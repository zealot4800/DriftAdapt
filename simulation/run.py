#!/usr/bin/env python3
"""Run the packet simulation and print one result line per method."""
import argparse
from pathlib import Path
from dotenv import load_dotenv
import yaml
if __package__:
    from .src.config import METHODS, load_config
    from .src.engine import run
else:
    from src.config import METHODS, load_config
    from src.engine import run


def main():
    parser = argparse.ArgumentParser(description='Packet simulation: CARAVAN, SiteGuard and DriftAdapt')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--method', choices=(*METHODS, 'all'), default='all')
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parent/'.env', override=False)
    try:
        results = run(load_config(args.config), args.method)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(1, f'error: {" ".join(str(exc).split())}\n')
    for name, result in results.items():
        if 'error' in result:
            print(f'{name}: error: {" ".join(result["error"].split())}')
            continue
        metrics = result['metrics']
        def score(key):
            value = metrics.get(key) if metrics else None
            return f'{value:.3f}' if value is not None else 'n/a'
        telemetry = result.get('reference')
        extra = (f" refs={telemetry['released']} bytes={telemetry['export_bytes']} status={result['repair_status']}"
                 if telemetry is not None else '')
        print(f"{name}: windows={result['windows']} F1={score('macro_f1')} "
              f"FPR={score('false_positive_rate')} recall={score('recall')} "
              f"updates={result['adaptation_count']} time={result['seconds']:.2f}s{extra}")
    if any('error' in result for result in results.values()):
        parser.exit(1)


if __name__ == '__main__':
    main()
