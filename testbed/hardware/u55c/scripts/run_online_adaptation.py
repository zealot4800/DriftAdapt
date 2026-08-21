#!/usr/bin/env python3
"""Run CARAVAN or DriftAdapt host-controlled adaptation against the U55C."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv


RECORD_PATTERN = re.compile(r"^DRIFTADAPT_WINDOW_RECORD=(.+)$", re.MULTILINE)
HIST_RECORD_PATTERN = re.compile(r"^DRIFTADAPT_HIST_RECORD=(.+)$", re.MULTILINE)
HIST_RANGE_PATTERN = re.compile(r"^DRIFTADAPT_HIST_RANGE=(.+)$", re.MULTILINE)
VALUE_PATTERN = re.compile(r"^DRIFTADAPT_([A-Z0-9_]+)=(.+)$", re.MULTILINE)
PARAMETER_KEYS = (
    "linear_relu_stack.0.weight",
    "linear_relu_stack.0.bias",
    "linear_relu_stack.2.weight",
    "linear_relu_stack.2.bias",
    "linear_relu_stack.4.weight",
    "linear_relu_stack.4.bias",
)
PARAMETER_COUNT = 182
AXIS_CLOCK_HZ = 250_000_000


def parse_integer(value: str) -> int:
    return int(value, 0)


def signed_16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def signed_32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def pack_bitmap(values: list[int], storage_words: int) -> list[int]:
    words = [0] * storage_words
    for index, value in enumerate(values):
        if value:
            words[index // 32] |= 1 << (index % 32)
    return words


def macro_f1_from_counts(tp: int, tn: int, fp: int, fn: int) -> float:
    positive = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    negative = (2 * tn) / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return (positive + negative) / 2


def fixed_parameter_words(model: torch.nn.Module) -> list[int]:
    state = model.state_dict()
    missing = [key for key in PARAMETER_KEYS if key not in state]
    if missing:
        raise RuntimeError(f"student model is missing parameters: {', '.join(missing)}")
    values = np.concatenate([
        state[key].detach().cpu().numpy().reshape(-1) for key in PARAMETER_KEYS
    ]).astype(np.float64)
    scaled = np.rint(values * (1 << 16)).astype(np.int64)
    if len(scaled) != PARAMETER_COUNT:
        raise RuntimeError(f"expected {PARAMETER_COUNT} parameters, received {len(scaled)}")
    if np.any(scaled < -(1 << 31)) or np.any(scaled > (1 << 31) - 1):
        raise RuntimeError("retrained model contains a parameter outside signed Q16.16")
    return [int(value) & 0xFFFFFFFF for value in scaled]


class JtagTransport:
    def __init__(self, *, vivado: Path, hw_server: str, script: Path) -> None:
        self.vivado = vivado
        self.hw_server = hw_server
        self.script = script
        if not vivado.is_file():
            raise FileNotFoundError(f"Vivado is unavailable: {vivado}")
        if not script.is_file():
            raise FileNotFoundError(f"JTAG control script is unavailable: {script}")

    def _run(self, *arguments: str) -> str:
        environment = os.environ.copy()
        environment.pop("LD_LIBRARY_PATH", None)
        environment.pop("PYTHONPATH", None)
        command = [
            str(self.vivado), "-mode", "batch", "-nojournal", "-nolog", "-notrace",
            "-source", str(self.script), "-tclargs", self.hw_server, *arguments,
        ]
        completed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=environment, check=False,
        )
        if completed.returncode:
            diagnostic = "\n".join(completed.stdout.splitlines()[-40:])
            raise RuntimeError(f"Vivado JTAG operation failed:\n{diagnostic}")
        return completed.stdout

    @staticmethod
    def _values(output: str) -> dict[str, str]:
        return {name: value.strip() for name, value in VALUE_PATTERN.findall(output)}

    def snapshot(self) -> dict[str, Any] | None:
        output = self._run("snapshot", "auto")
        values = self._values(output)
        if values.get("WINDOW_READY") == "0":
            if values.get("ONLINE_COMPLETE") == "1":
                return {"complete": True}
            return None
        required = {"WINDOW_BANK", "WINDOW_ID", "WINDOW_COUNT", "WINDOW_FIRST_SAMPLE"}
        missing = required - values.keys()
        if missing:
            raise RuntimeError(f"window snapshot omitted: {', '.join(sorted(missing))}")
        records = []
        for encoded in RECORD_PATTERN.findall(output):
            fields = encoded.strip().split(",")
            if len(fields) != 12:
                raise RuntimeError(f"malformed FPGA window record: {encoded}")
            record_number = int(fields[0], 10)
            words = [int(value, 16) for value in fields[1:]]
            feature_values: list[float] = []
            for word in words[:8]:
                feature_values.append(signed_16(word & 0xFFFF) / 256.0)
                feature_values.append(signed_16((word >> 16) & 0xFFFF) / 256.0)
            records.append({
                "record": record_number,
                "features": feature_values,
                "sample_id": words[8],
                "prediction": words[9] & 1,
                "margin_q16": words[10],
            })
        count = parse_integer(values["WINDOW_COUNT"])
        if len(records) != count:
            raise RuntimeError(f"FPGA advertised {count} samples but returned {len(records)}")
        records.sort(key=lambda row: row["record"])
        histogram_required = {
            "HIST_STATUS", "HIST_WINDOW_ID", "HIST_WINDOW_COUNT",
            "HIST_FEATURE_COUNT", "HIST_BIN_COUNT", "HIST_REFERENCE_WINDOWS",
            "HIST_STATE_BYTES",
        }
        histogram_missing = histogram_required - values.keys()
        if histogram_missing:
            raise RuntimeError(
                f"histogram snapshot omitted: {', '.join(sorted(histogram_missing))}"
            )
        feature_count = parse_integer(values["HIST_FEATURE_COUNT"])
        bin_count = parse_integer(values["HIST_BIN_COUNT"])
        reference_counts = [[0] * bin_count for _ in range(feature_count)]
        current_counts = [[0] * bin_count for _ in range(feature_count)]
        seen_counters: set[tuple[int, int]] = set()
        for encoded in HIST_RECORD_PATTERN.findall(output):
            fields = [int(value, 0) for value in encoded.strip().split(",")]
            if len(fields) != 4:
                raise RuntimeError(f"malformed FPGA histogram record: {encoded}")
            feature_index, bin_index, reference, current = fields
            if not (0 <= feature_index < feature_count and 0 <= bin_index < bin_count):
                raise RuntimeError(f"FPGA histogram selector is out of range: {encoded}")
            reference_counts[feature_index][bin_index] = reference
            current_counts[feature_index][bin_index] = current
            seen_counters.add((feature_index, bin_index))
        if len(seen_counters) != feature_count * bin_count:
            raise RuntimeError("FPGA returned an incomplete histogram snapshot")
        ranges = [[0, 0] for _ in range(feature_count)]
        for encoded in HIST_RANGE_PATTERN.findall(output):
            fields = [int(value, 0) for value in encoded.strip().split(",")]
            if len(fields) != 3 or not 0 <= fields[0] < feature_count:
                raise RuntimeError(f"malformed FPGA histogram range: {encoded}")
            ranges[fields[0]] = fields[1:]
        hist_status = parse_integer(values["HIST_STATUS"])
        histogram = {
            "window_id": parse_integer(values["HIST_WINDOW_ID"]),
            "window_count": parse_integer(values["HIST_WINDOW_COUNT"]),
            "feature_count": feature_count,
            "bin_count": bin_count,
            "reference_windows": parse_integer(values["HIST_REFERENCE_WINDOWS"]),
            "reference_ready": bool(hist_status & 0x2),
            "overflow_error": bool(hist_status & 0x8),
            "reference_counts": reference_counts,
            "current_counts": current_counts,
            "ranges_q8": ranges,
            "counter_count": 4 * feature_count * bin_count,
            "counter_bits": 32,
            "state_bytes": parse_integer(values["HIST_STATE_BYTES"]),
        }
        return {
            "bank": parse_integer(values["WINDOW_BANK"]),
            "window_id": parse_integer(values["WINDOW_ID"]),
            "count": count,
            "first_sample": parse_integer(values["WINDOW_FIRST_SAMPLE"]),
            "records": records,
            "histogram": histogram,
        }

    def submit_labels(
        self, bank: int, window_id: int, labels: list[int], window_size: int
    ) -> dict[str, int]:
        storage_words = (window_size + 31) // 32
        label_words = pack_bitmap(labels, storage_words)
        valid_words = pack_bitmap([1] * len(labels), storage_words)
        output = self._run(
            "submit", str(bank), str(window_id),
            ",".join(f"{word:08x}" for word in label_words),
            ",".join(f"{word:08x}" for word in valid_words),
        )
        values = self._values(output)
        names = ("WINDOW_ID", "COUNT", "TP", "TN", "FP", "FN", "MATCHES")
        result = {}
        for name in names:
            key = f"SUBMIT_{name}"
            if key not in values:
                raise RuntimeError(f"label submission omitted {key}")
            result[name.lower()] = parse_integer(values[key])
        return result

    @staticmethod
    def _update_metrics(values: dict[str, str]) -> dict[str, int]:
        names = (
            "TOTAL_PARAMETERS", "PATCHED_PARAMETERS", "BYTES", "CLONE_CYCLES",
            "PATCH_CYCLES", "COMMIT_CYCLES", "TOTAL_CYCLES",
            "OLD_MODEL_VERSION", "NEW_MODEL_VERSION",
        )
        metrics = {}
        for name in names:
            key = f"UPDATE_{name}"
            if key not in values:
                raise RuntimeError(f"hardware update omitted {key}")
            metrics[name.lower()] = parse_integer(values[key])
        return metrics

    def commit_weights(self, words: list[int]) -> dict[str, int]:
        with tempfile.NamedTemporaryFile("w", suffix=".mem", encoding="ascii") as stream:
            stream.write("".join(f"{word:08x}\n" for word in words))
            stream.flush()
            output = self._run("weights", stream.name)
        values = self._values(output)
        if values.get("WEIGHT_COMMIT") != "PASS":
            raise RuntimeError("FPGA did not acknowledge the atomic weight commit")
        metrics = self._update_metrics(values)
        if parse_integer(values["MODEL_VERSION"]) != metrics["new_model_version"]:
            raise RuntimeError("full update returned inconsistent model versions")
        return metrics

    def commit_selective(self, patches: list[tuple[int, int]]) -> dict[str, int]:
        if not patches:
            raise ValueError("selective hardware update requires at least one changed parameter")
        with tempfile.NamedTemporaryFile("w", suffix=".patch", encoding="ascii") as stream:
            stream.write("".join(f"{index},{value:08x}\n" for index, value in patches))
            stream.flush()
            output = self._run("selective", stream.name)
        values = self._values(output)
        if values.get("SELECTIVE_COMMIT") != "PASS":
            raise RuntimeError("FPGA did not acknowledge the selective atomic commit")
        metrics = self._update_metrics(values)
        if metrics["patched_parameters"] != len(patches):
            raise RuntimeError("FPGA selective patch count does not match the host request")
        if metrics["bytes"] != 4 * len(patches):
            raise RuntimeError("FPGA selective update byte count is inconsistent")
        return metrics


def resolve_config_path(value: str, base: Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def load_raw_feature_table(dataset_config: dict[str, Any], base: Path) -> torch.Tensor:
    """Load only agent-visible raw features; never read the ground-truth column."""
    feature_names = [name.strip().lower() for name in dataset_config["feature_names"]]
    frames = []
    for configured_path in dataset_config["files"]:
        path = Path(resolve_config_path(configured_path, base))
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if "total packet length" in feature_names and "total packet length" not in frame:
        frame["total packet length"] = (
            frame["total length of fwd packets"] + frame["total length of bwd packets"]
        )
    if "number of packets" in feature_names and "number of packets" not in frame:
        frame["number of packets"] = (
            frame["total fwd packets"] + frame["total backward packets"]
        )
    missing = [name for name in feature_names if name not in frame]
    if missing:
        raise RuntimeError(f"raw agent feature columns are missing: {', '.join(missing)}")
    return torch.tensor(frame[feature_names].to_numpy(dtype="float32"))


def main() -> None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[4]
    load_dotenv(repo_root / "stimulation/.env", override=False)
    sys.path.insert(0, str(repo_root))
    from stimulation.src.labelers import build_labeler
    from stimulation.src.drift import jensen_shannon_divergence
    from stimulation.src.impact import map_drift_to_neural_impact
    from stimulation.src.metrics import macro_f1
    from stimulation.src.models import build_model, load_checkpoint
    from stimulation.src.selector import select_drift_associated_samples
    from stimulation.src.trainer import (
        form_training_set, retrain_model, selective_retrain_model,
    )
    from stimulation.src.trigger import RetrainingTrigger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=repo_root / "stimulation/configs/cic-ids2017.yaml",
    )
    parser.add_argument(
        "--vivado", type=Path,
        default=Path(os.environ.get("VIVADO_BIN", "/home/palhad/Xilinx/Vivado/2022.2/bin/vivado")),
    )
    parser.add_argument("--hw-server", default="TCP:localhost:3121")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument(
        "--mode", choices=("caravan", "driftadapt-selective"),
        help="hardware update policy; defaults to driftadapt.mode from YAML",
    )
    parser.add_argument(
        "--provider", choices=("local", "dnn", "config"), default="local",
        help="local uses Ollama; dnn uses the checkpoint labeler; config preserves YAML type",
    )
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--max-windows", type=int, default=0, help="zero processes the full stream")
    parser.add_argument("--no-retrain", action="store_true", help="label and detect without updating weights")
    parser.add_argument(
        "--output", type=Path,
        default=repo_root / "testbed/results/online-windows.jsonl",
    )
    args = parser.parse_args()
    if args.poll_interval < 0 or args.max_windows < 0:
        parser.error("poll interval and max windows must be nonnegative")
    if not args.config.is_file():
        parser.error(f"configuration is unavailable: {args.config}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config_base = args.config.resolve().parent
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    configured_mode = str(config.get("driftadapt", {}).get("mode", "caravan"))
    configured_mode = configured_mode.replace("_", "-")
    if configured_mode == "baseline-caravan":
        configured_mode = "caravan"
    hardware_mode = args.mode or configured_mode
    if hardware_mode not in {"caravan", "driftadapt-selective"}:
        parser.error(f"unsupported hardware adaptation mode: {hardware_mode}")
    selective_mode = hardware_mode == "driftadapt-selective"
    seed = int(config.get("runtime", {}).get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    labeler_config = copy.deepcopy(config["labeler"])
    if args.provider == "local":
        labeler_config["type"] = "ollama"
    elif args.provider == "dnn":
        labeler_config["type"] = "dnn"
    labeler_config.setdefault("feature_names", config["dataset"]["feature_names"])
    if labeler_config.get("checkpoint"):
        labeler_config["checkpoint"] = resolve_config_path(
            labeler_config["checkpoint"], config_base,
        )
    labeler = build_labeler(labeler_config, device)
    raw_feature_table = (
        load_raw_feature_table(config["dataset"], config_base)
        if labeler_config["type"].lower() in {"llm", "ollama", "openai"}
        else None
    )

    model_config = config["model"]
    student = build_model(model_config["class"], model_config.get("kwargs")).to(device)
    load_checkpoint(
        student, resolve_config_path(model_config["checkpoint"], config_base), device,
    )
    student.eval()
    optimizer = None

    trigger_config = config["trigger"]
    if trigger_config.get("type") != "accuracy_proxy":
        parser.error("hardware adaptation requires trigger.type: accuracy_proxy")
    trigger = RetrainingTrigger(
        float(trigger_config["threshold"]),
        drop_threshold=float(trigger_config.get("drop_threshold", 0.15)),
        consecutive_windows=int(trigger_config.get("consecutive_windows", 2)),
    )
    training = config["training"]
    configured_window_size = int(config["stream"]["window_size"])
    drift_config = config["drift"]
    configured_bins = int(drift_config["bins"])
    drift_threshold = float(drift_config["threshold"])
    drift_consecutive_windows = int(drift_config["consecutive_windows"])
    drift_reference_windows = int(drift_config["reference_windows"])
    feature_names = list(config["dataset"]["feature_names"])
    if configured_bins != 16 or len(feature_names) != 16:
        parser.error("U55C histogram RTL requires exactly 16 features and drift.bins: 16")
    if not 0.0 <= drift_threshold <= 1.0:
        parser.error("drift.threshold must be between zero and one")
    if drift_consecutive_windows < 1 or drift_reference_windows < 1:
        parser.error("drift consecutive_windows and reference_windows must be positive")
    feature_drift_streaks = [0] * len(feature_names)
    impact_threshold = float(config.get("adaptation", {}).get("impact_threshold", 0.30))
    selection_config = config.get("sample_selection", {})

    transport = JtagTransport(
        vivado=args.vivado,
        hw_server=args.hw_server,
        script=script_path.with_name("driftadapt_online_control.tcl"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    adaptation_events = 0
    active_model_version = 0
    print("DRIFTADAPT_ONLINE_AGENT=READY", flush=True)

    while args.max_windows == 0 or processed < args.max_windows:
        snapshot = transport.snapshot()
        if snapshot is None:
            time.sleep(args.poll_interval)
            continue
        if snapshot.get("complete"):
            break
        if snapshot["count"] > configured_window_size:
            raise RuntimeError(
                f"FPGA window {snapshot['count']} exceeds configured size {configured_window_size}"
            )
        histogram = snapshot["histogram"]
        if histogram["window_id"] != snapshot["window_id"] or histogram["window_count"] != snapshot["count"]:
            raise RuntimeError("FPGA histogram and CARAVAN window snapshots are not aligned")
        if histogram["bin_count"] != configured_bins:
            raise RuntimeError("FPGA histogram bin count does not match drift.bins")
        if histogram["reference_windows"] != drift_reference_windows:
            raise RuntimeError("FPGA reference-window count does not match drift.reference_windows")
        if histogram["feature_count"] != len(feature_names):
            raise RuntimeError("FPGA histogram feature order does not match the dataset configuration")
        if histogram["overflow_error"]:
            raise RuntimeError("FPGA histogram monitor reported a counter overflow")

        reference_ready = bool(
            histogram["reference_ready"] and
            snapshot["window_id"] + 1 >= drift_reference_windows
        )
        feature_drift_scores: list[float | None]
        if reference_ready and snapshot["window_id"] >= drift_reference_windows:
            feature_drift_scores = []
            for feature_index in range(len(feature_names)):
                score = jensen_shannon_divergence(
                    torch.tensor(histogram["reference_counts"][feature_index], dtype=torch.int64),
                    torch.tensor(histogram["current_counts"][feature_index], dtype=torch.int64),
                )
                feature_drift_scores.append(score)
                if score > drift_threshold:
                    feature_drift_streaks[feature_index] = min(
                        feature_drift_streaks[feature_index] + 1,
                        drift_consecutive_windows,
                    )
                else:
                    feature_drift_streaks[feature_index] = 0
        else:
            feature_drift_scores = [None] * len(feature_names)
        drifted_feature_indices = [
            index for index, streak in enumerate(feature_drift_streaks)
            if streak >= drift_consecutive_windows
        ]
        hardware_feature_drift = bool(drifted_feature_indices)
        named_feature_scores = dict(zip(feature_names, feature_drift_scores))
        neural_impact = map_drift_to_neural_impact(
            student, named_feature_scores, impact_threshold,
        )
        impacted_modules = neural_impact["impact_selected_modules"]

        features = torch.tensor(
            [row["features"] for row in snapshot["records"]],
            dtype=torch.float32, device=device,
        )
        predictions = torch.tensor(
            [row["prediction"] for row in snapshot["records"]],
            dtype=torch.long, device=device,
        )
        labeling_started = time.perf_counter()
        if raw_feature_table is not None:
            sample_ids = [row["sample_id"] for row in snapshot["records"]]
            if any(index >= len(raw_feature_table) for index in sample_ids):
                raise RuntimeError("FPGA sample ID is outside the configured raw-feature source")
            raw_features = raw_feature_table[sample_ids].to(device)
        else:
            raw_features = features
        generated_labels = labeler.label(features, raw_features, None).to(
            device=device, dtype=torch.long,
        )
        labeling_time = time.perf_counter() - labeling_started
        if generated_labels.numel() != snapshot["count"] or not bool(
            torch.all((generated_labels == 0) | (generated_labels == 1))
        ):
            raise RuntimeError("local agent must return exactly one binary label per sample")

        proxy_f1 = macro_f1(generated_labels, predictions)
        labels_list = [int(value) for value in generated_labels.detach().cpu().tolist()]
        hardware = transport.submit_labels(
            snapshot["bank"], snapshot["window_id"], labels_list, configured_window_size
        )
        hardware_proxy = macro_f1_from_counts(
            hardware["tp"], hardware["tn"], hardware["fp"], hardware["fn"],
        )
        if hardware["window_id"] != snapshot["window_id"] or hardware["count"] != snapshot["count"]:
            raise RuntimeError("FPGA scored a different window than the local agent submitted")
        if abs(hardware_proxy - proxy_f1) > 1e-9:
            raise RuntimeError(
                f"host/FPGA accuracy proxy mismatch: host={proxy_f1}, fpga={hardware_proxy}"
            )

        drift = trigger.update(proxy_f1)
        adaptation_requested = hardware_feature_drift if selective_mode else drift
        retrained = False
        deployed = False
        training_time = 0.0
        candidate_proxy = proxy_f1
        model_version = active_model_version
        training_samples = 0
        hardware_update = {
            "total_parameters": PARAMETER_COUNT,
            "patched_parameters": 0,
            "bytes": 0,
            "clone_cycles": 0,
            "patch_cycles": 0,
            "commit_cycles": 0,
            "total_cycles": 0,
            "old_model_version": None,
            "new_model_version": None,
        }

        if adaptation_requested and not args.no_retrain:
            adaptation_features = features
            adaptation_labels = generated_labels
            if selective_mode:
                ranges = torch.tensor(
                    [[signed_32(value) / 256.0 for value in pair]
                     for pair in histogram["ranges_q8"]],
                    dtype=features.dtype, device=device,
                )
                selected_samples = select_drift_associated_samples(
                    features, named_feature_scores, drifted_feature_indices,
                    ranges[:, 0], ranges[:, 1],
                    enabled=bool(selection_config.get("enabled", False)),
                    max_samples=int(selection_config.get("max_samples", features.shape[0])),
                )
                if selected_samples.numel():
                    adaptation_features = features[selected_samples]
                    adaptation_labels = generated_labels[selected_samples]
            balanced = bool(training.get("balance_binary", True))
            train_x, train_y = form_training_set(
                adaptation_features, adaptation_labels, balance_binary=balanced,
            )
            if train_x is not None:
                before_state = copy.deepcopy(student.state_dict())
                before_words = fixed_parameter_words(student)
                training_samples = int(train_y.numel())
                if selective_mode:
                    training_time, _ = selective_retrain_model(
                        student, train_x, train_y, impacted_modules,
                        epochs=int(training["epochs"]),
                        batch_size=int(training["batch_size"]),
                        learning_rate=float(training["learning_rate"]),
                        optimizer_name=training.get("optimizer", "Adam"),
                    )
                    optimizer = None
                else:
                    training_time, optimizer = retrain_model(
                        student, train_x, train_y,
                        epochs=int(training["epochs"]),
                        batch_size=int(training["batch_size"]),
                        learning_rate=float(training["learning_rate"]),
                        optimizer_name=training.get("optimizer", "Adam"),
                        optimizer=optimizer,
                    )
                retrained = True
                with torch.inference_mode():
                    candidate_labels = student.run_inference(features)[1]
                candidate_proxy = macro_f1(generated_labels, candidate_labels)
                if candidate_proxy >= proxy_f1:
                    new_words = fixed_parameter_words(student)
                    if selective_mode:
                        patches = [
                            (index, value) for index, value in enumerate(new_words)
                            if value != before_words[index]
                        ]
                        if patches:
                            hardware_update = transport.commit_selective(patches)
                            deployed = True
                    else:
                        hardware_update = transport.commit_weights(new_words)
                        deployed = True
                    if deployed:
                        model_version = hardware_update["new_model_version"]
                        active_model_version = model_version
                        checkpoint_path = args.output.with_name("online-student.pt")
                        torch.save(student.state_dict(), checkpoint_path)
                    else:
                        student.load_state_dict(before_state)
                        student.eval()
                else:
                    student.load_state_dict(before_state)
                    student.eval()
                    optimizer = None

        if deployed:
            adaptation_events += 1

        row: dict[str, Any] = {
            "window_id": snapshot["window_id"],
            "bank": snapshot["bank"],
            "first_sample": snapshot["first_sample"],
            "sample_count": snapshot["count"],
            "proxy_macro_f1": proxy_f1,
            "matches": hardware["matches"],
            "true_positive": hardware["tp"],
            "true_negative": hardware["tn"],
            "false_positive": hardware["fp"],
            "false_negative": hardware["fn"],
            "drift": drift,
            "hardware_mode": hardware_mode,
            "hardware_adaptation_triggered": adaptation_requested,
            "adaptation_event": deployed,
            "adaptation_event_count": adaptation_events,
            "labeling_time_seconds": labeling_time,
            "retrained": retrained,
            "training_samples": training_samples,
            "training_time_seconds": training_time,
            "candidate_proxy_macro_f1": candidate_proxy,
            "deployed": deployed,
            "model_version": model_version,
            "feature_drift_reference_ready": reference_ready,
            "feature_drift_scores": named_feature_scores,
            "drifted_feature_indices": drifted_feature_indices,
            "drifted_feature_names": [feature_names[index] for index in drifted_feature_indices],
            "number_of_drifted_features": len(drifted_feature_indices),
            "feature_drift_detected": hardware_feature_drift,
            "histogram_counter_count": histogram["counter_count"],
            "histogram_counter_bits": histogram["counter_bits"],
            "histogram_state_bytes": histogram["state_bytes"],
            "impact_selected_modules": impacted_modules,
            "hardware_total_parameters": hardware_update["total_parameters"],
            "hardware_patched_parameters": hardware_update["patched_parameters"],
            "hardware_patched_percentage": (
                100.0 * hardware_update["patched_parameters"] / PARAMETER_COUNT
            ),
            "hardware_transferred_bytes": hardware_update["bytes"],
            "hardware_clone_latency_cycles": hardware_update["clone_cycles"],
            "hardware_clone_latency_seconds": hardware_update["clone_cycles"] / AXIS_CLOCK_HZ,
            "hardware_parameter_transfer_latency_cycles": hardware_update["patch_cycles"],
            "hardware_parameter_transfer_latency_seconds": (
                hardware_update["patch_cycles"] / AXIS_CLOCK_HZ
            ),
            "hardware_commit_latency_cycles": hardware_update["commit_cycles"],
            "hardware_commit_latency_seconds": hardware_update["commit_cycles"] / AXIS_CLOCK_HZ,
            "hardware_total_update_latency_cycles": hardware_update["total_cycles"],
            "hardware_total_update_latency_seconds": (
                hardware_update["total_cycles"] / AXIS_CLOCK_HZ
            ),
            "hardware_old_model_version": hardware_update["old_model_version"],
            "hardware_new_model_version": hardware_update["new_model_version"],
        }
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        processed += 1
        print(
            f"window={snapshot['window_id']} samples={snapshot['count']} "
            f"proxy_macro_f1={proxy_f1:.6f} drift={int(drift)} "
            f"feature_drift={int(hardware_feature_drift)} "
            f"mode={hardware_mode} patches={hardware_update['patched_parameters']} "
            f"retrained={int(retrained)} deployed={int(deployed)}",
            flush=True,
        )

    print(f"DRIFTADAPT_ONLINE_WINDOWS={processed}", flush=True)
    print(f"DRIFTADAPT_ADAPTATION_EVENTS={adaptation_events}", flush=True)


if __name__ == "__main__":
    main()
