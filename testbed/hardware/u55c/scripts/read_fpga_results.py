#!/usr/bin/env python3
"""Read DRIFTADAPT online-window progress, proxy quality, and performance over JTAG."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


FEATURE_WORD = 0x44524654
FORMAT_VERSION = 0x00040001
RAW_PATTERN = re.compile(r"^DRIFTADAPT_JTAG_([A-Z0-9_]+)=0x([0-9a-fA-F]+)$", re.MULTILINE)
U64_NAMES = (
    "captured", "labeled", "classified", "elapsed", "busy", "input_stall",
    "output_stall", "latency_sum", "latency_min", "latency_max", "sent",
    "total_cycles", "closed_loop",
)


def divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def combine_u64(raw: dict[str, int], name: str) -> int:
    key = name.upper()
    return (raw[f"{key}_HI"] << 32) | raw[f"{key}_LO"]


def run_jtag_reader(args: argparse.Namespace, control_script: Path) -> dict[str, int]:
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            str(args.vivado), "-mode", "batch", "-nojournal", "-nolog", "-notrace",
            "-source", str(control_script), "-tclargs", args.hw_server,
            "status", str(max(0.0, args.wait)),
        ],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=environment, check=False,
    )
    if completed.returncode:
        diagnostic = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(f"Vivado JTAG read failed:\n{diagnostic}")
    raw = {name: int(value, 16) for name, value in RAW_PATTERN.findall(completed.stdout)}
    required = {
        "FEATURE", "STATUS", "VERSION", "DATASET_COUNT", "CLOCK_HZ", "MODE",
        "WINDOW_SIZE", "READY_MASK", "LAST_WINDOW_ID", "LAST_WINDOW_COUNT",
        "LAST_TP", "LAST_TN", "LAST_FP", "LAST_FN", "LAST_MATCHES",
        "WINDOWS_LABELED", "MODEL_VERSION", "WEIGHTS_STAGED", "WEIGHT_STATUS",
    }
    required.update(f"{name.upper()}_{half}" for name in U64_NAMES for half in ("LO", "HI"))
    missing = sorted(required - raw.keys())
    if missing:
        raise RuntimeError(f"JTAG output omitted registers: {', '.join(missing)}")
    return raw


def calculate_report(raw: dict[str, int]) -> tuple[dict[str, int | float | str], bool]:
    if raw["FEATURE"] != FEATURE_WORD:
        raise RuntimeError(f"DRIFTADAPT feature word mismatch: 0x{raw['FEATURE']:08x}")
    if raw["VERSION"] != FORMAT_VERSION:
        raise RuntimeError(
            f"DRIFTADAPT ABI mismatch: 0x{raw['VERSION']:08x}; expected 0x{FORMAT_VERSION:08x}"
        )
    if raw["MODE"] != 2:
        raise RuntimeError(f"unsupported DRIFTADAPT mode: {raw['MODE']}")

    counters = {name: combine_u64(raw, name) for name in U64_NAMES}
    tp, tn, fp, fn = (raw[name] for name in ("LAST_TP", "LAST_TN", "LAST_FP", "LAST_FN"))
    positive_f1 = divide(2 * tp, 2 * tp + fp + fn)
    negative_f1 = divide(2 * tn, 2 * tn + fp + fn)
    proxy_macro_f1 = (positive_f1 + negative_f1) / 2
    classified = counters["classified"]
    elapsed = counters["elapsed"]
    closed_loop = counters["closed_loop"]
    clock_hz = raw["CLOCK_HZ"]
    throughput_pps = divide(classified * clock_hz, closed_loop)
    inference_throughput_pps = divide(classified * clock_hz, counters["busy"])
    cycle_ns = divide(1_000_000_000.0, clock_hz)
    mean_latency_cycles = divide(counters["latency_sum"], classified)
    online_complete = bool(raw["STATUS"] & (1 << 3))

    report: dict[str, int | float | str] = {
        "status": raw["STATUS"],
        "register_version": raw["VERSION"],
        "mode": "fpga_inference_local_agent",
        "generator_active": (raw["STATUS"] >> 1) & 1,
        "generator_finished": (raw["STATUS"] >> 2) & 1,
        "online_complete": int(online_complete),
        "window0_ready": (raw["READY_MASK"] >> 0) & 1,
        "window1_ready": (raw["READY_MASK"] >> 1) & 1,
        "scoring_active": (raw["STATUS"] >> 6) & 1,
        "label_error": (raw["STATUS"] >> 7) & 1,
        "weights_loaded": (raw["STATUS"] >> 8) & 1,
        "weight_commit_pending": (raw["STATUS"] >> 9) & 1,
        "dataset_count": raw["DATASET_COUNT"],
        "window_size": raw["WINDOW_SIZE"],
        "windows_labeled": raw["WINDOWS_LABELED"],
        "captured": counters["captured"],
        "labeled": counters["labeled"],
        "classified": classified,
        "last_window_id": raw["LAST_WINDOW_ID"],
        "last_window_count": raw["LAST_WINDOW_COUNT"],
        "last_true_positive": tp,
        "last_true_negative": tn,
        "last_false_positive": fp,
        "last_false_negative": fn,
        "last_matches": raw["LAST_MATCHES"],
        "last_proxy_macro_f1": proxy_macro_f1,
        "model_version": raw["MODEL_VERSION"],
        "weights_staged": raw["WEIGHTS_STAGED"],
        "active_weight_bank": (raw["WEIGHT_STATUS"] >> 10) & 1,
        "clock_hz": clock_hz,
        "dataplane_elapsed_cycles": elapsed,
        "dataplane_elapsed_seconds": divide(elapsed, clock_hz),
        "closed_loop_cycles": closed_loop,
        "closed_loop_seconds": divide(closed_loop, clock_hz),
        "closed_loop_throughput_samples_per_second": throughput_pps,
        "classifier_throughput_samples_per_second": inference_throughput_pps,
        "feature_payload_gbps": throughput_pps * 32 * 8 / 1e9,
        "classifier_utilization": divide(counters["busy"], elapsed),
        "input_stall_fraction": divide(counters["input_stall"], elapsed),
        "output_stall_fraction": divide(counters["output_stall"], elapsed),
        "mean_latency_cycles": mean_latency_cycles,
        "mean_latency_ns": mean_latency_cycles * cycle_ns,
        "min_latency_ns": counters["latency_min"] * cycle_ns,
        "max_latency_ns": counters["latency_max"] * cycle_ns,
    }
    passed = (
        online_complete
        and counters["sent"] == raw["DATASET_COUNT"]
        and counters["captured"] == raw["DATASET_COUNT"]
        and counters["labeled"] == raw["DATASET_COUNT"]
        and classified == raw["DATASET_COUNT"]
        and not report["label_error"]
        and not report["weight_commit_pending"]
    )
    report["DRIFTADAPT_FPGA_RESULT"] = "PASS" if passed else "INCOMPLETE"
    return report, passed


def print_report(report: dict[str, int | float | str]) -> None:
    for name, value in report.items():
        if name in {"status", "register_version"}:
            print(f"{name}=0x{int(value):08x}")
        elif isinstance(value, float):
            print(f"{name}={value:.9f}")
        else:
            print(f"{name}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vivado", type=Path,
        default=Path(os.environ.get("VIVADO_BIN", "/home/palhad/Xilinx/Vivado/2022.2/bin/vivado")),
    )
    parser.add_argument("--hw-server", default="TCP:localhost:3121")
    parser.add_argument("--wait", type=float, default=0.0, metavar="SECONDS")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.vivado.is_file():
        parser.error(f"Vivado is unavailable: {args.vivado}")
    try:
        raw = run_jtag_reader(args, Path(__file__).with_name("driftadapt_online_control.tcl"))
        report, passed = calculate_report(raw)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
