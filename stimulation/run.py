#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

if __package__:
    from .src.pipeline import DriftAdaptPipeline
else:
    from src.pipeline import DriftAdaptPipeline


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env", override=False)

    parser = argparse.ArgumentParser(description="Run the minimal DRIFTADAPT baseline")
    parser.add_argument("--config", required=True, type=Path)
    provider = parser.add_mutually_exclusive_group()
    provider.add_argument(
        "-local", "--local", dest="provider", action="store_const", const="local",
        help="use the local Ollama labeler",
    )
    provider.add_argument(
        "-cloud", "--cloud", dest="provider", action="store_const", const="cloud",
        help="use the OpenAI Responses API labeler",
    )
    parser.add_argument(
        "--adaptation-mode", choices=("full", "selective"),
        help="legacy alias: full selects CARAVAN, selective selects DriftAdapt selective",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline_caravan", "driftadapt_full", "driftadapt_selective"),
        help="override driftadapt.mode from the YAML configuration",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.provider:
        config["labeler"]["type"] = "ollama" if args.provider == "local" else "openai"
        # Keep results from explicit local and cloud runs separate.
        config["dataset"]["name"] = f'{config["dataset"]["name"]}-{args.provider}'
    if args.adaptation_mode:
        config.setdefault("driftadapt", {})["mode"] = (
            "baseline_caravan" if args.adaptation_mode == "full" else "driftadapt_selective"
        )
    if args.mode:
        config.setdefault("driftadapt", {})["mode"] = args.mode
    _, summary = DriftAdaptPipeline(config, config_path).run()
    labels = {
        "dataset": "Dataset", "number_of_windows": "Number of windows",
        "number_of_retraining_events": "Number of retraining events",
        "average_student_f1": "Average student F1", "average_static_f1": "Average static F1",
        "average_labeler_f1": "Average labeler F1", "total_retraining_time": "Total retraining time",
        "average_retraining_time": "Average retraining time",
        "number_of_drift_events": "Number of drift events",
        "number_of_adaptation_events": "Number of adaptation events",
        "total_model_update_bytes": "Total model update bytes",
    }
    for key, label in labels.items():
        value = summary[key]
        print(f"{label}: {value:.6f}" if isinstance(value, float) else f"{label}: {value}")


if __name__ == "__main__":
    main()
