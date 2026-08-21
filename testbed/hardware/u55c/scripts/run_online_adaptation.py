#!/usr/bin/env python3
"""Run CARAVAN-style labeling, drift detection, and retraining against the U55C."""

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


def parse_integer(value: str) -> int:
    return int(value, 0)


def signed_16(value: int) -> int:
    return value - 0x10000 if value & 0x8000 else value


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
        return {
            "bank": parse_integer(values["WINDOW_BANK"]),
            "window_id": parse_integer(values["WINDOW_ID"]),
            "count": count,
            "first_sample": parse_integer(values["WINDOW_FIRST_SAMPLE"]),
            "records": records,
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

    def commit_weights(self, words: list[int]) -> int:
        with tempfile.NamedTemporaryFile("w", suffix=".mem", encoding="ascii") as stream:
            stream.write("".join(f"{word:08x}\n" for word in words))
            stream.flush()
            output = self._run("weights", stream.name)
        values = self._values(output)
        if values.get("WEIGHT_COMMIT") != "PASS":
            raise RuntimeError("FPGA did not acknowledge the atomic weight commit")
        return parse_integer(values["MODEL_VERSION"])


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
    from stimulation.src.metrics import macro_f1
    from stimulation.src.models import build_model, load_checkpoint
    from stimulation.src.trainer import form_training_set, retrain_model
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

    transport = JtagTransport(
        vivado=args.vivado,
        hw_server=args.hw_server,
        script=script_path.with_name("driftadapt_online_control.tcl"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
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
        retrained = False
        deployed = False
        training_time = 0.0
        candidate_proxy = proxy_f1
        model_version: int | None = None
        training_samples = 0

        if drift and not args.no_retrain:
            balanced = bool(training.get("balance_binary", True))
            train_x, train_y = form_training_set(
                features, generated_labels, balance_binary=balanced,
            )
            if train_x is not None:
                before_state = copy.deepcopy(student.state_dict())
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
                    model_version = transport.commit_weights(fixed_parameter_words(student))
                    deployed = True
                    checkpoint_path = args.output.with_name("online-student.pt")
                    torch.save(student.state_dict(), checkpoint_path)
                else:
                    student.load_state_dict(before_state)
                    student.eval()
                    optimizer = None

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
            "labeling_time_seconds": labeling_time,
            "retrained": retrained,
            "training_samples": training_samples,
            "training_time_seconds": training_time,
            "candidate_proxy_macro_f1": candidate_proxy,
            "deployed": deployed,
            "model_version": model_version,
        }
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        processed += 1
        print(
            f"window={snapshot['window_id']} samples={snapshot['count']} "
            f"proxy_macro_f1={proxy_f1:.6f} drift={int(drift)} "
            f"retrained={int(retrained)} deployed={int(deployed)}",
            flush=True,
        )

    print(f"DRIFTADAPT_ONLINE_WINDOWS={processed}", flush=True)


if __name__ == "__main__":
    main()
