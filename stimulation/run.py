#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

if __package__:
    from .src import create_pipeline, normalize_runtime_mode
else:
    from src import create_pipeline, normalize_runtime_mode


def _parse_runtime_mode(value: str) -> str:
    try:
        return normalize_runtime_mode(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env", override=False)

    parser = argparse.ArgumentParser(
        description="Run CARAVAN, DriftAdapt or SiteGuard simulation"
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
        metavar="{caravan,driftadapt,drift_conditioned,siteguard}",
        help=(
            "override architecture or legacy driftadapt.mode; the deprecated baseline_caravan alias "
            "is accepted for compatibility"
        ),
    )
    parser.add_argument('--action', choices=['audit', 'train', 'infer', 'adapt'],
                        help='SiteGuard action; defaults to siteguard.action in the dataset YAML')
    parser.add_argument('--packets', type=Path, help='SiteGuard packet CSV override (relative to cwd)')
    parser.add_argument('--checkpoint', type=Path, help='SiteGuard checkpoint override (relative to cwd)')
    parser.add_argument('--feedback', type=Path, help='SiteGuard verified-feedback JSONL override')
    parser.add_argument('--output', type=Path, help='SiteGuard run output directory override')
    parser.add_argument('--verified-benign-source', action='store_true',
                        help='explicitly attest that SiteGuard training input is verified benign')
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.mode is None:
        try:
            runtime_mode = normalize_runtime_mode(
                config.get('architecture', config.get("driftadapt", {}).get("mode", "caravan"))
            )
        except ValueError as exc:
            parser.error(f"invalid architecture/mode in {config_path}: {exc}")
    else:
        runtime_mode = args.mode
    if args.provider and runtime_mode != "caravan":
        parser.error("--local and --cloud apply only to --mode caravan")
    siteguard_options = (args.action, args.packets, args.checkpoint, args.feedback,
                         args.output, args.verified_benign_source)
    if runtime_mode != 'siteguard' and any(siteguard_options):
        parser.error('--action/--packets/--checkpoint/--feedback/--output/--verified-benign-source apply only to siteguard')
    if args.provider:
        config["labeler"]["type"] = "ollama" if args.provider == "local" else "openai"
        # Keep results from explicit local and cloud runs separate.
        config["dataset"]["name"] = f'{config["dataset"]["name"]}-{args.provider}'
    config['architecture'] = runtime_mode
    if runtime_mode == 'siteguard':
        section = config.setdefault('siteguard', {})
        action = args.action or section.get('action', 'infer')
        if args.feedback and action != 'adapt':
            parser.error('--feedback requires --action adapt')
        if args.verified_benign_source and action != 'train':
            parser.error('--verified-benign-source requires --action train')
        section['action'] = action
        if args.packets:
            section['source_packets' if action == 'train' else 'evaluation_packets'] = str(args.packets.resolve())
        for name in ('checkpoint', 'feedback', 'output'):
            value = getattr(args, name)
            if value:
                section[name] = str(value.resolve())
        if args.verified_benign_source:
            section['verified_benign_source'] = True
    try:
        _, summary = create_pipeline(config, config_path).run()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    if runtime_mode == 'siteguard':
        print(json.dumps(summary, indent=2, allow_nan=False))
        return
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
