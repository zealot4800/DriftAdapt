#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import yaml
from dotenv import load_dotenv

if __package__:
    from .src.pipeline import DriftAdaptPipeline
else:
    from src.pipeline import DriftAdaptPipeline


def _normalize_runtime_mode(value: str) -> str:
    """Normalize the two public modes and the deprecated CARAVAN alias."""
    mode = str(value).strip().lower()
    if mode == "caravan":
        return mode
    if mode == "drift_conditioned":
        return mode
    if mode == "baseline_caravan":
        warnings.warn(
            "'baseline_caravan' is deprecated; use '--mode caravan' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return "caravan"
    raise ValueError("mode must be 'caravan' or 'drift_conditioned'")


def _parse_runtime_mode(value: str) -> str:
    try:
        return _normalize_runtime_mode(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env", override=False)

    parser = argparse.ArgumentParser(
        description="Run CARAVAN or drift-conditioned DriftAdapt simulation"
    )
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
        "--mode",
        type=_parse_runtime_mode,
        metavar="{caravan,drift_conditioned}",
        help=(
            "override driftadapt.mode; the deprecated baseline_caravan alias "
            "is accepted for compatibility"
        ),
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.mode is None:
        try:
            runtime_mode = _normalize_runtime_mode(
                config.get("driftadapt", {}).get("mode", "caravan")
            )
        except ValueError as exc:
            parser.error(f"invalid driftadapt.mode in {config_path}: {exc}")
    else:
        runtime_mode = args.mode
    if args.provider and runtime_mode == "drift_conditioned":
        parser.error("--local and --cloud apply only to --mode caravan")
    if args.provider:
        config["labeler"]["type"] = "ollama" if args.provider == "local" else "openai"
        # Keep results from explicit local and cloud runs separate.
        config["dataset"]["name"] = f'{config["dataset"]["name"]}-{args.provider}'
    config.setdefault("driftadapt", {})["mode"] = runtime_mode
    _, summary = DriftAdaptPipeline(config, config_path).run()
    labels = {
        "method": "Method",
        "average_f1": "Average F1",
        "post_drift_f1": "Post-drift F1",
        "number_of_adaptation_events": "Number of adaptation events",
        "total_teacher_samples": "Total teacher samples",
        "total_labeling_time": "Total labeling time",
        "total_retraining_time": "Total retraining time",
        "total_adaptation_recovery_time": "Total adaptation/recovery time",
        "average_adaptation_latency": "Average adaptation latency",
        "total_updated_parameters": "Total updated parameters",
        "average_parameter_update_percentage": "Average parameter update %",
        "total_model_update_bytes": "Total model-update bytes",
    }
    for key, label in labels.items():
        value = summary[key]
        print(f"{label}: {value:.6f}" if isinstance(value, float) else f"{label}: {value}")


if __name__ == "__main__":
    main()
