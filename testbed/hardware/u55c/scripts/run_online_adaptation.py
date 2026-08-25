#!/usr/bin/env python3
"""Run CARAVAN or drift-conditioned adaptation against the U55C."""

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
import warnings

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
PUBLIC_MODES = {"caravan", "drift_conditioned"}


def normalize_mode(value: str) -> str:
    """Normalize the two public modes and the deprecated CARAVAN alias."""
    mode = str(value).strip().lower().replace("-", "_")
    if mode in PUBLIC_MODES:
        return mode
    if mode == "baseline_caravan":
        warnings.warn(
            "'baseline_caravan' is deprecated; use 'caravan' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return "caravan"
    raise ValueError("hardware mode must be 'caravan' or 'drift_conditioned'")


def parse_mode(value: str) -> str:
    try:
        return normalize_mode(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


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


def changed_parameter_patches(
    active_words: list[int], candidate_words: list[int],
) -> list[tuple[int, int]]:
    """Return exact changed indices in the fixed 182-word hardware layout."""
    if len(active_words) != PARAMETER_COUNT or len(candidate_words) != PARAMETER_COUNT:
        raise ValueError(f"hardware models must contain exactly {PARAMETER_COUNT} words")
    return [
        (index, candidate & 0xFFFFFFFF)
        for index, (active, candidate) in enumerate(zip(active_words, candidate_words))
        if (active & 0xFFFFFFFF) != (candidate & 0xFFFFFFFF)
    ]


def histogram_equivalence(
    features: torch.Tensor, histogram: dict[str, Any], *, enabled: bool,
) -> dict[str, Any]:
    """Reconstruct the fixed-bin snapshot for host/RTL equivalence reporting."""
    if not enabled:
        return {
            "histogram_equivalence_checked": False,
            "histogram_equivalent": None,
            "histogram_mismatched_counters": 0,
        }
    values_q8 = torch.round(features.detach().cpu() * 256.0).to(torch.int64)
    ranges = [
        (signed_32(pair[0]), signed_32(pair[1]))
        for pair in histogram["ranges_q8"]
    ]
    bins = int(histogram["bin_count"])
    reconstructed = [[0] * bins for _ in ranges]
    for row in values_q8.tolist():
        for feature_index, value in enumerate(row):
            minimum, maximum = ranges[feature_index]
            if maximum <= minimum or value <= minimum:
                bin_index = 0
            elif value >= maximum:
                bin_index = bins - 1
            else:
                bin_index = min(
                    bins - 1,
                    ((value - minimum) * bins) // (maximum - minimum),
                )
            reconstructed[feature_index][bin_index] += 1
    current = histogram["current_counts"]
    mismatches = sum(
        reconstructed[feature][bin_index] != current[feature][bin_index]
        for feature in range(len(ranges))
        for bin_index in range(bins)
    )
    return {
        "histogram_equivalence_checked": True,
        "histogram_equivalent": mismatches == 0,
        "histogram_mismatched_counters": mismatches,
    }


def finite_mean(values: list[float | None]) -> float | None:
    selected = [float(value) for value in values if value is not None and np.isfinite(value)]
    return sum(selected) / len(selected) if selected else None


def empty_hardware_update() -> dict[str, int | None]:
    return {
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


def load_embedded_parameter_words(path: Path) -> list[int]:
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) != PARAMETER_COUNT:
        raise RuntimeError(
            f"embedded hardware model must contain {PARAMETER_COUNT} parameters"
        )
    try:
        return [int(line.strip(), 16) & 0xFFFFFFFF for line in lines]
    except ValueError as error:
        raise RuntimeError(f"embedded hardware model is malformed: {path}") from error


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
        required = {
            "WINDOW_BANK", "WINDOW_ID", "WINDOW_COUNT", "WINDOW_FIRST_SAMPLE",
            "WINDOW_MODEL_VERSION", "ACTIVE_MODEL_VERSION",
        }
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
            "inference_model_version": parse_integer(values["WINDOW_MODEL_VERSION"]),
            "active_model_version": parse_integer(values["ACTIVE_MODEL_VERSION"]),
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

    def release_window(self, bank: int, window_id: int) -> dict[str, int]:
        """Release a completed window without writing or scoring any labels."""
        output = self._run("release", str(bank), str(window_id))
        values = self._values(output)
        result: dict[str, int] = {}
        for name in ("BANK", "WINDOW_ID", "COUNT"):
            key = f"RELEASE_{name}"
            if key not in values:
                raise RuntimeError(f"label-free window release omitted {key}")
            result[name.lower()] = parse_integer(values[key])
        if result["bank"] != bank or result["window_id"] != window_id:
            raise RuntimeError("FPGA released a different window than the host requested")
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
        if len(words) != PARAMETER_COUNT:
            raise ValueError(
                f"full hardware update requires exactly {PARAMETER_COUNT} words"
            )
        with tempfile.NamedTemporaryFile("w", suffix=".mem", encoding="ascii") as stream:
            stream.write("".join(f"{word:08x}\n" for word in words))
            stream.flush()
            output = self._run("weights", stream.name)
        values = self._values(output)
        if values.get("WEIGHT_COMMIT") != "PASS":
            raise RuntimeError("FPGA did not acknowledge the atomic weight commit")
        metrics = self._update_metrics(values)
        if metrics["total_parameters"] != PARAMETER_COUNT:
            raise RuntimeError("FPGA full update reported an invalid model size")
        if metrics["patched_parameters"] != PARAMETER_COUNT:
            raise RuntimeError("FPGA full update did not stage every model parameter")
        if metrics["bytes"] != 4 * PARAMETER_COUNT:
            raise RuntimeError("FPGA full update byte count is inconsistent")
        if parse_integer(values["MODEL_VERSION"]) != metrics["new_model_version"]:
            raise RuntimeError("full update returned inconsistent model versions")
        if metrics["new_model_version"] != metrics["old_model_version"] + 1:
            raise RuntimeError("full update did not advance the model version once")
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
        if metrics["total_parameters"] != PARAMETER_COUNT:
            raise RuntimeError("FPGA selective update reported an invalid model size")
        if metrics["patched_parameters"] != len(patches):
            raise RuntimeError("FPGA selective patch count does not match the host request")
        if metrics["bytes"] != 4 * len(patches):
            raise RuntimeError("FPGA selective update byte count is inconsistent")
        if parse_integer(values["MODEL_VERSION"]) != metrics["new_model_version"]:
            raise RuntimeError("selective update returned inconsistent model versions")
        if metrics["new_model_version"] != metrics["old_model_version"] + 1:
            raise RuntimeError("selective update did not advance the model version once")
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
    from stimulation.src.drift import jensen_shannon_divergence
    from stimulation.src.models import build_model, load_checkpoint
    from drift_conditioned_runtime import (
        DriftConditionedRuntime,
        apply_parameter_patches,
        fixed_inference,
    )

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
        "--mode", type=parse_mode, metavar="{caravan,drift_conditioned}",
        help=(
            "hardware policy; defaults to driftadapt.mode from YAML; "
            "baseline_caravan remains a deprecated alias"
        ),
    )
    parser.add_argument(
        "--provider", choices=("local", "dnn", "config"), default=None,
        help=(
            "CARAVAN-only labeler override: local uses Ollama, dnn uses the "
            "checkpoint labeler, and config preserves the YAML type"
        ),
    )
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--max-windows", type=int, default=0, help="zero processes the full stream")
    parser.add_argument(
        "--no-retrain", action="store_true",
        help="observe windows without CARAVAN retraining or DriftAdapt repair deployment",
    )
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
    if args.mode is not None:
        hardware_mode = args.mode
    else:
        try:
            hardware_mode = normalize_mode(
                config.get("driftadapt", {}).get("mode", "caravan")
            )
        except ValueError as error:
            parser.error(str(error))
    conditioned_mode = hardware_mode == "drift_conditioned"
    if conditioned_mode and args.provider is not None:
        parser.error("--provider applies only to --mode caravan")
    seed = int(config.get("runtime", {}).get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_config = config["model"]
    student = build_model(model_config["class"], model_config.get("kwargs")).to(device)
    load_checkpoint(
        student, resolve_config_path(model_config["checkpoint"], config_base), device,
    )
    student.eval()
    embedded_weight_path = repo_root / "testbed/hardware/u55c/assets/driftadapt_weights.mem"
    active_parameter_words = fixed_parameter_words(student)
    if active_parameter_words != load_embedded_parameter_words(embedded_weight_path):
        raise RuntimeError(
            "configured student checkpoint does not match the 182-parameter model "
            "embedded in the U55C image; regenerate FPGA assets and rebuild"
        )
    model_words_by_version: dict[int, tuple[int, ...]] = {
        0: tuple(active_parameter_words),
    }
    pending_equivalence_versions: set[int] = set()
    validated_model_versions: set[int] = set()
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

    labeler = None
    raw_feature_table = None
    trigger = None
    optimizer = None
    training: dict[str, Any] | None = None
    conditioned_runtime = None
    if conditioned_mode:
        conditioned_runtime = DriftConditionedRuntime.from_config(
            config,
            config_base,
            student,
            device,
            initial_parameter_words=active_parameter_words,
        )
    else:
        from stimulation.src.labelers import build_labeler
        from stimulation.src.metrics import macro_f1
        from stimulation.src.trainer import form_training_set, retrain_model
        from stimulation.src.trigger import RetrainingTrigger

        labeler_config = copy.deepcopy(config["labeler"])
        if args.provider == "local":
            labeler_config["type"] = "ollama"
        elif args.provider == "dnn":
            labeler_config["type"] = "dnn"
        labeler_config.setdefault("feature_names", feature_names)
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
        trigger_config = config["trigger"]
        if trigger_config.get("type") != "accuracy_proxy":
            parser.error("CARAVAN requires trigger.type: accuracy_proxy")
        trigger = RetrainingTrigger(
            float(trigger_config["threshold"]),
            drop_threshold=float(trigger_config.get("drop_threshold", 0.15)),
            consecutive_windows=int(trigger_config.get("consecutive_windows", 2)),
        )
        training = config["training"]

    transport = JtagTransport(
        vivado=args.vivado,
        hw_server=args.hw_server,
        script=script_path.with_name("driftadapt_online_control.tcl"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    adaptation_events = 0
    active_model_version = 0
    last_healthy_proxy: float | None = None
    output_rows: list[dict[str, Any]] = []
    stream_completed = False
    print("DRIFTADAPT_ONLINE_AGENT=READY", flush=True)

    while args.max_windows == 0 or processed < args.max_windows:
        collection_started = time.perf_counter()
        snapshot = transport.snapshot()
        window_collection_latency = time.perf_counter() - collection_started
        if snapshot is None:
            time.sleep(args.poll_interval)
            continue
        if snapshot.get("complete"):
            stream_completed = True
            break
        if snapshot["active_model_version"] != active_model_version:
            raise RuntimeError(
                "host student and active FPGA model versions diverged: "
                f"host={active_model_version}, fpga={snapshot['active_model_version']}"
            )
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
            histogram["reference_ready"]
            and snapshot["window_id"] >= drift_reference_windows
        )
        drift_detection_started = time.perf_counter()
        feature_drift_scores: list[float | None]
        if (
            not conditioned_mode
            and reference_ready
            and snapshot["window_id"] >= drift_reference_windows
        ):
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
        drift_detection_latency = time.perf_counter() - drift_detection_started

        features = torch.tensor(
            [row["features"] for row in snapshot["records"]],
            dtype=torch.float32, device=device,
        )
        predictions = torch.tensor(
            [row["prediction"] for row in snapshot["records"]],
            dtype=torch.long, device=device,
        )
        inference_version = int(snapshot["inference_model_version"])
        if inference_version not in model_words_by_version:
            raise RuntimeError(
                f"window references unknown model version {inference_version}"
            )
        fixed_outputs = fixed_inference(
            features, model_words_by_version[inference_version],
        )
        software_predictions = torch.tensor(
            [output.prediction for output in fixed_outputs],
            dtype=torch.long,
            device=device,
        )
        software_margins = [output.margin_q16 for output in fixed_outputs]
        hardware_margins = [int(row["margin_q16"]) for row in snapshot["records"]]
        prediction_matches = software_predictions == predictions
        margin_matches = [
            software == hardware
            for software, hardware in zip(software_margins, hardware_margins)
        ]
        software_hardware_match_count = int(prediction_matches.sum().item())
        software_hardware_mismatch_indices = torch.nonzero(
            ~prediction_matches, as_tuple=False,
        ).squeeze(1).detach().cpu().tolist()
        software_hardware_margin_mismatch_indices = [
            index for index, matches in enumerate(margin_matches) if not matches
        ]
        exact_fixed_equivalence = (
            software_hardware_match_count == snapshot["count"]
            and not software_hardware_margin_mismatch_indices
        )
        if not exact_fixed_equivalence:
            raise RuntimeError(
                "FPGA model failed exact fixed-point prediction/margin equivalence: "
                f"version={inference_version}"
            )
        validated_model_versions.add(inference_version)
        if inference_version in pending_equivalence_versions:
            pending_equivalence_versions.remove(inference_version)
        histogram_check = histogram_equivalence(
            features,
            histogram,
            enabled=reference_ready and snapshot["window_id"] >= drift_reference_windows,
        )

        proxy_f1: float | None = None
        hardware_proxy: float | None = None
        accuracy_proxy_triggered = False
        labeling_time = 0.0
        teacher_samples = 0
        conditioned_update = None
        adapter_metrics: dict[str, Any] = {
            "adapter_prediction_attempted": False,
            "adapter_confidence": None,
            "adapter_confidence_threshold": (
                conditioned_runtime.confidence_threshold if conditioned_mode else None
            ),
            "adapter_confidence_passed": None,
            "adapter_rejection_reason": None,
            "adapter_module_probabilities": {},
            "predicted_modules": [],
            "predicted_module_magnitudes": {},
            "drift_signature_vector": [],
            "drift_signature_per_feature": {},
            "drift_signature_fields": [],
            "drift_signature_format_version": None,
            "drift_signature_size": 0,
            "repair_candidate_accepted": None,
            "repair_patched_modules": [],
            "repair_parameter_indices": [],
            "repair_parameter_format": None,
            "repair_changed_parameters": 0,
            "repair_changed_parameter_percentage": 0.0,
            "repair_software_changed_parameters": 0,
            "repair_software_changed_parameter_percentage": 0.0,
            "repair_update_bytes": 0,
            "signature_latency_seconds": 0.0,
            "adapter_prediction_latency_seconds": 0.0,
            "repair_application_latency_seconds": 0.0,
            "repair_latency_seconds": 0.0,
        }
        hardware_score: dict[str, int | None] = {
            "window_id": snapshot["window_id"],
            "count": snapshot["count"],
            "tp": None,
            "tn": None,
            "fp": None,
            "fn": None,
            "matches": None,
        }
        released_without_labels = False
        if conditioned_mode:
            conditioned_update = conditioned_runtime.propose(histogram, features)
            named_feature_scores = dict(conditioned_update.feature_drift_scores)
            feature_drift_scores = [
                named_feature_scores[name] for name in feature_names
            ]
            drifted_feature_indices = list(conditioned_update.drifted_feature_indices)
            hardware_feature_drift = bool(conditioned_update.feature_drift_detected)
            drift_detection_latency = float(
                conditioned_update.drift_detection_latency_seconds
            )
            signature_per_feature = dict(conditioned_update.signature_per_feature)
            adapter_metrics.update({
                "adapter_prediction_attempted": (
                    conditioned_update.adapter_confidence is not None
                ),
                "adapter_confidence": conditioned_update.adapter_confidence,
                "adapter_confidence_passed": conditioned_update.confidence_passed,
                "adapter_rejection_reason": conditioned_update.rejection_reason,
                "adapter_module_probabilities": dict(
                    conditioned_update.module_probabilities
                ),
                "predicted_modules": list(conditioned_update.predicted_modules),
                "predicted_module_magnitudes": dict(
                    conditioned_update.module_magnitudes
                ),
                "drift_signature_vector": list(conditioned_update.signature_vector),
                "drift_signature_per_feature": signature_per_feature,
                "drift_signature_fields": list(conditioned_update.signature_fields),
                "drift_signature_format_version": int(
                    conditioned_update.signature_format_version
                ),
                "drift_signature_size": len(conditioned_update.signature_vector),
                "repair_candidate_accepted": bool(conditioned_update.accepted),
                "repair_patched_modules": list(conditioned_update.patched_modules),
                "repair_parameter_indices": [
                    int(index) for index, _value in conditioned_update.patches
                ],
                "repair_parameter_format": conditioned_update.parameter_format,
                "repair_changed_parameters": int(
                    conditioned_update.changed_parameter_count
                ),
                "repair_changed_parameter_percentage": float(
                    conditioned_update.changed_parameter_percentage
                ),
                "repair_software_changed_parameters": int(
                    conditioned_update.software_changed_parameter_count
                ),
                "repair_software_changed_parameter_percentage": float(
                    conditioned_update.software_changed_parameter_percentage
                ),
                "repair_update_bytes": int(conditioned_update.update_bytes),
                "signature_latency_seconds": float(
                    conditioned_update.signature_latency_seconds
                ),
                "adapter_prediction_latency_seconds": float(
                    conditioned_update.adapter_latency_seconds
                ),
                "repair_application_latency_seconds": float(
                    conditioned_update.repair_latency_seconds
                ),
                "repair_latency_seconds": float(conditioned_update.latency_seconds),
            })
            adaptation_requested = bool(
                hardware_feature_drift
                and conditioned_update.rejection_reason
                != "drift_event_already_handled"
            )
            if args.no_retrain and conditioned_update.accepted:
                conditioned_runtime.discard()
        else:
            labeling_started = time.perf_counter()
            if raw_feature_table is not None:
                sample_ids = [row["sample_id"] for row in snapshot["records"]]
                if any(index >= len(raw_feature_table) for index in sample_ids):
                    raise RuntimeError(
                        "FPGA sample ID is outside the configured raw-feature source"
                    )
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
            teacher_samples = snapshot["count"]
            proxy_f1 = macro_f1(generated_labels, predictions)
            labels_list = [int(value) for value in generated_labels.detach().cpu().tolist()]
            hardware_score = transport.submit_labels(
                snapshot["bank"], snapshot["window_id"], labels_list,
                configured_window_size,
            )
            hardware_proxy = macro_f1_from_counts(
                hardware_score["tp"], hardware_score["tn"],
                hardware_score["fp"], hardware_score["fn"],
            )
            if (
                hardware_score["window_id"] != snapshot["window_id"]
                or hardware_score["count"] != snapshot["count"]
            ):
                raise RuntimeError(
                    "FPGA scored a different window than the local agent submitted"
                )
            if abs(hardware_proxy - proxy_f1) > 1e-9:
                raise RuntimeError(
                    f"host/FPGA accuracy proxy mismatch: host={proxy_f1}, "
                    f"fpga={hardware_proxy}"
                )
            accuracy_proxy_triggered = bool(trigger.update(proxy_f1))
            if not accuracy_proxy_triggered and proxy_f1 >= trigger.threshold:
                last_healthy_proxy = proxy_f1
            adaptation_requested = accuracy_proxy_triggered

        retrained = False
        deployed = False
        training_time = 0.0
        candidate_proxy = proxy_f1
        model_version = active_model_version
        training_samples = 0
        hardware_update = empty_hardware_update()
        hardware_update_wall_time = 0.0
        parameter_reconstruction_mismatches = 0
        updated_parameter_indices: list[int] = []

        if adaptation_requested and not args.no_retrain:
            if conditioned_mode:
                patches = [
                    (int(index), int(value) & 0xFFFFFFFF)
                    for index, value in conditioned_update.patches
                ]
                if len({index for index, _ in patches}) != len(patches):
                    raise RuntimeError("drift-conditioned repair contains duplicate indices")
                if any(index < 0 or index >= PARAMETER_COUNT for index, _ in patches):
                    raise RuntimeError("drift-conditioned repair index is outside 0..181")
                if any(active_parameter_words[index] == value for index, value in patches):
                    raise RuntimeError("drift-conditioned repair contains an unchanged patch")
                if len(patches) != int(conditioned_update.changed_parameter_count):
                    raise RuntimeError("adapter and host changed-parameter counts disagree")
                if conditioned_update.accepted and patches:
                    candidate_words = conditioned_runtime.candidate_parameter_words(
                        conditioned_update
                    )
                    reconstructed_words = apply_parameter_patches(
                        active_parameter_words, patches,
                    )
                    parameter_reconstruction_mismatches = sum(
                        active != candidate
                        for active, candidate in zip(
                            reconstructed_words, candidate_words,
                        )
                    )
                    if parameter_reconstruction_mismatches:
                        raise RuntimeError(
                            "sparse repair does not exactly reconstruct the quantized candidate"
                        )
                    hardware_update_started = time.perf_counter()
                    hardware_update = transport.commit_selective(patches)
                    hardware_update_wall_time = (
                        time.perf_counter() - hardware_update_started
                    )
                    if hardware_update["old_model_version"] != active_model_version:
                        raise RuntimeError("selective repair started from the wrong model version")
                    model_version = int(hardware_update["new_model_version"])
                    if model_version != active_model_version + 1:
                        raise RuntimeError(
                            "selective repair did not advance the model version exactly once"
                        )
                    conditioned_runtime.accept(conditioned_update)
                    if conditioned_runtime.active_parameter_words != candidate_words:
                        raise RuntimeError(
                            "committed adapter state differs from the patched parameter image"
                        )
                    active_parameter_words = list(candidate_words)
                    updated_parameter_indices = [index for index, _value in patches]
                    student = conditioned_runtime.active_model
                    model_words_by_version[model_version] = candidate_words
                    pending_equivalence_versions.add(model_version)
                    deployed = True
                    active_model_version = model_version
                    torch.save(
                        student.state_dict(), args.output.with_name("online-student.pt")
                    )
            else:
                balanced = bool(training.get("balance_binary", True))
                train_x, train_y = form_training_set(
                    features, generated_labels, balance_binary=balanced,
                )
                if train_x is None:
                    train_y = None
                else:
                    before_state = copy.deepcopy(student.state_dict())
                    new_words = active_parameter_words
                    training_samples = int(train_y.numel())
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
                        hardware_update_started = time.perf_counter()
                        hardware_update = transport.commit_weights(new_words)
                        hardware_update_wall_time = (
                            time.perf_counter() - hardware_update_started
                        )
                        if hardware_update["old_model_version"] != active_model_version:
                            raise RuntimeError("full update started from the wrong model version")
                        deployed = True
                        model_version = int(hardware_update["new_model_version"])
                        if model_version != active_model_version + 1:
                            raise RuntimeError(
                                "full update did not advance the model version exactly once"
                            )
                        active_model_version = model_version
                        active_parameter_words = new_words
                        updated_parameter_indices = list(range(PARAMETER_COUNT))
                        parameter_reconstruction_mismatches = 0
                        model_words_by_version[model_version] = tuple(new_words)
                        pending_equivalence_versions.add(model_version)
                        torch.save(
                            student.state_dict(), args.output.with_name("online-student.pt")
                        )
                    else:
                        student.load_state_dict(before_state)
                        student.eval()
                        optimizer = None

        if conditioned_mode:
            released = transport.release_window(
                snapshot["bank"], snapshot["window_id"],
            )
            if released["count"] != snapshot["count"]:
                raise RuntimeError("FPGA label-free release returned the wrong sample count")
            released_without_labels = True

        if deployed:
            adaptation_events += 1

        controller_processing_latency = (
            drift_detection_latency
            + adapter_metrics["signature_latency_seconds"]
            + adapter_metrics["adapter_prediction_latency_seconds"]
            + adapter_metrics["repair_application_latency_seconds"]
            if conditioned_mode else
            labeling_time + training_time
        )
        adaptation_latency = (
            window_collection_latency
            + controller_processing_latency
            + hardware_update_wall_time
        ) if deployed else 0.0
        hardware_programming_latency = (
            float(hardware_update["total_cycles"]) / AXIS_CLOCK_HZ
        )
        controller_to_hardware_latency = max(
            hardware_update_wall_time - hardware_programming_latency,
            0.0,
        )
        pre_drift_f1 = (
            last_healthy_proxy if deployed and not conditioned_mode else None
        )
        post_drift_f1 = proxy_f1 if deployed and not conditioned_mode else None
        post_adaptation_f1 = (
            candidate_proxy if deployed and not conditioned_mode else None
        )
        recovery_ratio = None
        if (
            pre_drift_f1 is not None
            and post_drift_f1 is not None
            and post_adaptation_f1 is not None
            and pre_drift_f1 - post_drift_f1 > 1e-12
        ):
            recovery_ratio = (
                (post_adaptation_f1 - post_drift_f1)
                / (pre_drift_f1 - post_drift_f1)
            )

        row: dict[str, Any] = {
            "record_type": "window",
            "window_id": snapshot["window_id"],
            "bank": snapshot["bank"],
            "first_sample": snapshot["first_sample"],
            "sample_count": snapshot["count"],
            "method": hardware_mode,
            "student_f1": proxy_f1,
            "static_f1": None,
            "proxy_macro_f1": proxy_f1,
            "hardware_proxy_macro_f1": hardware_proxy,
            "matches": hardware_score["matches"],
            "true_positive": hardware_score["tp"],
            "true_negative": hardware_score["tn"],
            "false_positive": hardware_score["fp"],
            "false_negative": hardware_score["fn"],
            "drift": (
                hardware_feature_drift if conditioned_mode
                else accuracy_proxy_triggered
            ),
            "accuracy_proxy_triggered": accuracy_proxy_triggered,
            "hardware_mode": hardware_mode,
            "hardware_adaptation_triggered": adaptation_requested,
            "adaptation_event": deployed,
            "adaptation_event_count": adaptation_events,
            "teacher_samples": teacher_samples,
            "labeling_time_seconds": labeling_time,
            "retrained": retrained,
            "training_samples": training_samples,
            "training_time_seconds": training_time,
            "retraining_time_seconds": training_time,
            "candidate_proxy_macro_f1": candidate_proxy,
            "f1_before_adaptation": proxy_f1 if deployed else None,
            "f1_after_adaptation": candidate_proxy if deployed else None,
            "model_f1_before_drift": pre_drift_f1,
            "f1_after_drift": post_drift_f1,
            "f1_after_predicted_repair": post_adaptation_f1,
            "oracle_retrained_f1": None,
            "recovery_ratio": recovery_ratio,
            "f1_unavailable_reason": (
                "feature-only hardware stream contains no runtime ground truth"
                if conditioned_mode else None
            ),
            "deployed": deployed,
            "adaptation_latency_seconds": adaptation_latency,
            "window_collection_latency_seconds": window_collection_latency,
            "statistics_collection_latency_seconds": (
                window_collection_latency if conditioned_mode else None
            ),
            "label_collection_latency_seconds": (
                None if conditioned_mode else window_collection_latency
            ),
            "controller_processing_latency_seconds": controller_processing_latency,
            "model_version": model_version,
            "inference_model_version": snapshot["inference_model_version"],
            "released_without_labels": released_without_labels,
            "feature_drift_reference_ready": reference_ready,
            "feature_drift_scores": named_feature_scores,
            "drifted_feature_indices": drifted_feature_indices,
            "drifted_feature_names": [feature_names[index] for index in drifted_feature_indices],
            "number_of_drifted_features": len(drifted_feature_indices),
            "feature_drift_detected": hardware_feature_drift,
            "feature_drift_detection_latency_seconds": drift_detection_latency,
            "histogram_counter_count": histogram["counter_count"],
            "histogram_counter_bits": histogram["counter_bits"],
            "histogram_state_bytes": histogram["state_bytes"],
            "impact_selected_modules": adapter_metrics["predicted_modules"],
            "parameters_updated": hardware_update["patched_parameters"],
            "updated_parameter_indices": updated_parameter_indices,
            "updated_parameter_format": "signed Q16.16",
            "parameter_update_percentage": (
                100.0 * int(hardware_update["patched_parameters"]) / PARAMETER_COUNT
            ),
            "model_update_bytes": hardware_update["bytes"],
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
            "hardware_update_wall_time_seconds": hardware_update_wall_time,
            "controller_to_hardware_update_latency_seconds": (
                controller_to_hardware_latency
            ),
            "hardware_programming_latency_seconds": hardware_programming_latency,
            "hardware_old_model_version": hardware_update["old_model_version"],
            "hardware_new_model_version": hardware_update["new_model_version"],
            "software_hardware_prediction_match_count": software_hardware_match_count,
            "software_hardware_prediction_match_percentage": (
                100.0 * software_hardware_match_count / snapshot["count"]
                if snapshot["count"] else 0.0
            ),
            "software_hardware_prediction_mismatch_indices": (
                software_hardware_mismatch_indices
            ),
            "software_hardware_margin_match_count": (
                snapshot["count"] - len(software_hardware_margin_mismatch_indices)
            ),
            "software_hardware_margin_match_percentage": (
                100.0
                * (snapshot["count"] - len(software_hardware_margin_mismatch_indices))
                / snapshot["count"]
                if snapshot["count"] else 0.0
            ),
            "software_hardware_margin_mismatch_indices": (
                software_hardware_margin_mismatch_indices
            ),
            "fixed_equivalence_prediction_tolerance": 0,
            "fixed_equivalence_margin_tolerance_q16": 0,
            "exact_fixed_equivalence": exact_fixed_equivalence,
            "parameter_mismatches": parameter_reconstruction_mismatches,
            **histogram_check,
            **adapter_metrics,
        }
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        output_rows.append(row)
        processed += 1
        proxy_display = "n/a" if proxy_f1 is None else f"{proxy_f1:.6f}"
        print(
            f"window={snapshot['window_id']} samples={snapshot['count']} "
            f"proxy_macro_f1={proxy_display} drift={int(accuracy_proxy_triggered)} "
            f"feature_drift={int(hardware_feature_drift)} "
            f"mode={hardware_mode} patches={hardware_update['patched_parameters']} "
            f"retrained={int(retrained)} deployed={int(deployed)}",
            flush=True,
        )

    event_rows = [row for row in output_rows if row["adaptation_event"]]
    histogram_rows = [
        row for row in output_rows if row["histogram_equivalence_checked"]
    ]
    pending_versions = sorted(pending_equivalence_versions)
    summary = {
        "record_type": "summary",
        "method": hardware_mode,
        "number_of_windows": processed,
        "stream_completed": stream_completed,
        "valid": not pending_versions,
        "average_f1": finite_mean([
            row["student_f1"] for row in output_rows
        ]),
        "pre_drift_f1": finite_mean([
            row["model_f1_before_drift"] for row in event_rows
        ]),
        "post_drift_f1": finite_mean([
            row["f1_after_drift"] for row in event_rows
        ]),
        "post_adaptation_f1": finite_mean([
            row["f1_after_predicted_repair"] for row in event_rows
        ]),
        "recovery_ratio": finite_mean([
            row["recovery_ratio"] for row in event_rows
        ]),
        "f1_unavailable_reason": (
            "feature-only hardware stream contains no runtime ground truth"
            if conditioned_mode else None
        ),
        "average_proxy_macro_f1": finite_mean([
            row["proxy_macro_f1"] for row in output_rows
        ]),
        "number_of_adaptation_events": adaptation_events,
        "total_teacher_samples": sum(row["teacher_samples"] for row in output_rows),
        "total_labeling_time_seconds": sum(
            row["labeling_time_seconds"] for row in output_rows
        ),
        "total_retraining_time_seconds": sum(
            row["retraining_time_seconds"] for row in output_rows
        ),
        "total_adaptation_recovery_time_seconds": sum(
            row["adaptation_latency_seconds"] for row in event_rows
        ),
        "average_adaptation_latency_seconds": finite_mean([
            row["adaptation_latency_seconds"] for row in event_rows
        ]) or 0.0,
        "total_window_collection_latency_seconds": sum(
            row["window_collection_latency_seconds"] for row in event_rows
        ),
        "total_controller_processing_latency_seconds": sum(
            row["controller_processing_latency_seconds"] for row in event_rows
        ),
        "total_updated_parameters": sum(
            row["parameters_updated"] for row in event_rows
        ),
        "average_parameter_update_percentage": finite_mean([
            row["parameter_update_percentage"] for row in event_rows
        ]) or 0.0,
        "total_model_update_bytes": sum(
            row["model_update_bytes"] for row in event_rows
        ),
        "total_hardware_update_latency_seconds": sum(
            row["hardware_total_update_latency_seconds"] for row in event_rows
        ),
        "total_hardware_update_wall_time_seconds": sum(
            row["hardware_update_wall_time_seconds"] for row in event_rows
        ),
        "total_controller_to_hardware_update_latency_seconds": sum(
            row["controller_to_hardware_update_latency_seconds"]
            for row in event_rows
        ),
        "total_hardware_programming_latency_seconds": sum(
            row["hardware_programming_latency_seconds"] for row in event_rows
        ),
        "total_feature_drift_detection_latency_seconds": sum(
            row["feature_drift_detection_latency_seconds"] for row in output_rows
        ),
        "total_signature_latency_seconds": sum(
            row["signature_latency_seconds"] for row in output_rows
        ),
        "total_adapter_prediction_latency_seconds": sum(
            row["adapter_prediction_latency_seconds"] for row in output_rows
        ),
        "total_repair_application_latency_seconds": sum(
            row["repair_application_latency_seconds"] for row in output_rows
        ),
        "final_model_version": active_model_version,
        "software_hardware_prediction_match_percentage": finite_mean([
            row["software_hardware_prediction_match_percentage"] for row in output_rows
        ]),
        "histogram_equivalence_checks": len(histogram_rows),
        "histogram_equivalence_failures": sum(
            row["histogram_equivalent"] is False for row in histogram_rows
        ),
        "exact_fixed_equivalence": all(
            row["exact_fixed_equivalence"] for row in output_rows
        ),
        "parameter_mismatches": sum(
            row["parameter_mismatches"] or 0 for row in output_rows
        ),
        "validated_model_versions": sorted(validated_model_versions),
        "pending_model_versions": pending_versions,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(f"DRIFTADAPT_ONLINE_WINDOWS={processed}", flush=True)
    print(f"DRIFTADAPT_ADAPTATION_EVENTS={adaptation_events}", flush=True)
    print(f"DRIFTADAPT_SUMMARY={json.dumps(summary, sort_keys=True)}", flush=True)
    if pending_versions:
        raise RuntimeError(
            "deployed model versions were never observed for exact hardware/software "
            f"equivalence: {pending_versions}"
        )


if __name__ == "__main__":
    main()
