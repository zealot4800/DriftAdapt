#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


def main() -> None:
    stimulation_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stimulation_root))
    from src.adapter import evaluate_adapter, save_adapter_bundle, train_drift_adapter
    from src.data import resolve_path
    from src.drift_dataset import analyze_drift_repair_learnability, load_drift_repair_dataset
    from src.drift_signature import DRIFT_SIGNATURE_FORMAT_VERSION, SIGNATURE_FIELDS
    from src.models import build_model, load_checkpoint
    from src.repair import model_fingerprint

    parser = argparse.ArgumentParser(description="Train and evaluate the drift adapter")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--module-threshold", type=float, default=0.50)
    parser.add_argument("--minimum-similarity-correlation", type=float, default=0.30)
    parser.add_argument("--minimum-module-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-recovery-ratio", type=float, default=0.50)
    parser.add_argument("--minimum-oracle-improvement-rate", type=float, default=0.50)
    args = parser.parse_args()
    if args.hidden_size < 1 or args.epochs < 1 or args.learning_rate <= 0:
        parser.error("adapter hidden size, epochs, and learning rate must be positive")
    if not 0.0 <= args.module_threshold <= 1.0:
        parser.error("--module-threshold must be between zero and one")
    if not -1.0 <= args.minimum_similarity_correlation <= 1.0:
        parser.error("--minimum-similarity-correlation must be between -1 and 1")
    for name, value in (
        ("--minimum-module-accuracy", args.minimum_module_accuracy),
        ("--minimum-recovery-ratio", args.minimum_recovery_ratio),
        ("--minimum-oracle-improvement-rate", args.minimum_oracle_improvement_rate),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be between zero and one")

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config.get("runtime", {}).get("seed", 42))
    artifact = load_drift_repair_dataset(args.dataset.resolve())
    model_config = config["model"]
    model = build_model(model_config["class"], model_config.get("kwargs"))
    load_checkpoint(
        model, str(resolve_path(model_config["checkpoint"], config_path.parent)),
        torch.device("cpu"),
    )
    expected_signature_config = {
        "bins": int(config.get("drift", {}).get("bins", 16)),
        "reference_windows": int(config.get("drift", {}).get("reference_windows", 10)),
        "normalized_shift_clip": float(
            config.get("drift_conditioned", {}).get("signature_shift_clip", 8.0)
        ),
    }
    if artifact["model_fingerprint"] != model_fingerprint(model):
        raise ValueError("Drift-repair dataset was generated for different model weights")
    if artifact["feature_names"] != list(config["dataset"]["feature_names"]):
        raise ValueError("Drift-repair dataset feature ordering does not match the config")
    if artifact["signature_config"] != expected_signature_config:
        raise ValueError("Drift-repair dataset signature settings do not match the config")
    if (
        artifact["signature_format_version"] != DRIFT_SIGNATURE_FORMAT_VERSION
        or artifact["signature_fields"] != list(SIGNATURE_FIELDS)
    ):
        raise ValueError("Drift-repair dataset signature format is unsupported")
    bundle, training_report = train_drift_adapter(
        artifact, hidden_size=args.hidden_size, epochs=args.epochs,
        learning_rate=args.learning_rate, seed=seed,
    )
    evaluation = evaluate_adapter(
        artifact, bundle, model, module_threshold=args.module_threshold,
        confidence_threshold=float(
            config.get("drift_conditioned", {}).get("confidence_threshold", 0.50)
        ),
        parameter_bits=int(config.get("driftadapt", {}).get("parameter_bits", 32)),
    )
    learnability = analyze_drift_repair_learnability(artifact)
    held_out = [
        metrics for split, metrics in evaluation["splits"].items()
        if split != "train"
    ]
    held_out_module_accuracy = min(
        (metrics["module_prediction_accuracy"] for metrics in held_out
         if metrics["module_prediction_accuracy"] is not None),
        default=None,
    )
    held_out_recovery = min(
        (metrics["recovery_ratio"] for metrics in held_out
         if metrics["recovery_ratio"] is not None),
        default=None,
    )
    held_out_improvement_rate = min(
        (metrics["oracle_improvement_rate"] for metrics in held_out),
        default=None,
    )
    held_out_repair_coverage = min(
        (metrics["repair_coverage"] for metrics in held_out),
        default=None,
    )
    held_out_similarity_correlation = learnability["held_out"][
        "signature_repair_similarity_correlation"
    ]
    validation = {
        "minimum_similarity_correlation": args.minimum_similarity_correlation,
        "minimum_module_accuracy": args.minimum_module_accuracy,
        "minimum_recovery_ratio": args.minimum_recovery_ratio,
        "minimum_oracle_improvement_rate": args.minimum_oracle_improvement_rate,
        "evaluated_confidence_threshold": float(
            config.get("drift_conditioned", {}).get("confidence_threshold", 0.50)
        ),
        "evaluated_module_threshold": args.module_threshold,
        "held_out_signature_repair_similarity_correlation": (
            held_out_similarity_correlation
        ),
        "held_out_module_prediction_accuracy": held_out_module_accuracy,
        "held_out_recovery_ratio": held_out_recovery,
        "held_out_oracle_improvement_rate": held_out_improvement_rate,
        "held_out_repair_coverage": held_out_repair_coverage,
    }
    validation["scientific_validation_passed"] = bool(
        held_out_similarity_correlation is not None
        and held_out_similarity_correlation >= args.minimum_similarity_correlation
        and held_out_module_accuracy is not None
        and held_out_module_accuracy >= args.minimum_module_accuracy
        and held_out_recovery is not None
        and held_out_recovery >= args.minimum_recovery_ratio
        and held_out_improvement_rate is not None
        and held_out_improvement_rate >= args.minimum_oracle_improvement_rate
    )
    report_payload = {
        "learnability": learnability,
        "training": training_report,
        "evaluation": evaluation,
        "scientific_validation": validation,
    }
    output = args.output or stimulation_root / "checkpoint" / (
        f"{config['dataset']['name']}-drift-adapter.pt"
    )
    report = args.report or output.with_suffix(".report.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(report_payload, indent=2, allow_nan=True) + "\n", encoding="utf-8",
    )
    print(f"Scientific report: {report}")
    if not validation["scientific_validation_passed"]:
        raise SystemExit("Scientific validation failed; adapter checkpoint was not written")
    save_adapter_bundle(bundle, output, {
        "training": training_report,
        "scientific_validation": validation,
        "signature_repair_similarity_correlation": learnability[
            "signature_repair_similarity_correlation"
        ],
    })
    print(f"Drift adapter: {output}")


if __name__ == "__main__":
    main()
