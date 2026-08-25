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
    from src.data import StreamingDataset, resolve_path
    from src.drift_dataset import (
        analyze_drift_repair_learnability,
        controlled_drift_scenarios,
        generate_drift_repair_dataset,
        save_drift_repair_dataset,
    )
    from src.models import build_model, load_checkpoint

    parser = argparse.ArgumentParser(description="Generate offline drift/oracle-repair data")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--severities", default="0.5,1.0,2.0")
    parser.add_argument("--directions", default="-1,1")
    parser.add_argument("--max-feature-pairs", type=int, default=16)
    parser.add_argument("--oracle-epochs", type=int, default=10)
    parser.add_argument("--module-threshold", type=float, default=0.80)
    args = parser.parse_args()
    if args.oracle_epochs < 1:
        parser.error("--oracle-epochs must be positive")
    if args.max_feature_pairs < 0:
        parser.error("--max-feature-pairs must be nonnegative")
    if not 0.0 <= args.module_threshold <= 1.0:
        parser.error("--module-threshold must be between zero and one")

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config.get("runtime", {}).get("seed", 42))
    torch.manual_seed(seed)
    device = torch.device("cpu")
    dataset_config = dict(config["dataset"])
    dataset = StreamingDataset(dataset_config, config_path.parent, device)
    windows = list(dataset.windows(int(config["stream"]["window_size"])))
    model_config = config["model"]
    model = build_model(model_config["class"], model_config.get("kwargs")).to(device)
    load_checkpoint(
        model, str(resolve_path(model_config["checkpoint"], config_path.parent)), device,
    )
    severities = [float(value) for value in args.severities.split(",") if value.strip()]
    directions = tuple(int(value) for value in args.directions.split(",") if value.strip())
    scenarios = controlled_drift_scenarios(
        len(dataset_config["feature_names"]), severities,
        max_feature_pairs=args.max_feature_pairs, directions=directions,
    )
    artifact = generate_drift_repair_dataset(
        model, windows, list(dataset_config["feature_names"]), config["drift"],
        config["training"], scenarios, seed=seed,
        oracle_epochs=args.oracle_epochs, module_threshold=args.module_threshold,
        signature_shift_clip=float(
            config.get("drift_conditioned", {}).get("signature_shift_clip", 8.0)
        ),
    )
    output = args.output or stimulation_root / "results" / (
        f"{dataset_config['name']}-drift-repair.pt"
    )
    report = args.report or output.with_suffix(".learnability.json")
    save_drift_repair_dataset(artifact, output)
    analysis = analyze_drift_repair_learnability(artifact)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(analysis, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"Drift-repair dataset: {output}")
    print(f"Learnability report: {report}")
    print(f"Scenarios: {len(artifact['records'])}")


if __name__ == "__main__":
    main()
