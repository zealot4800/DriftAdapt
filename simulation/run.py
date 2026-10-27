#!/usr/bin/env python3
"""Unified simulation runner."""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
if __package__:
    from .src.config import METHODS, PHASES, load_config
    from .src.engine import run
else:
    from src.config import METHODS, PHASES, load_config
    from src.engine import run


def main():
    parser = argparse.ArgumentParser(description='Independent CARAVAN, DriftAdapt and SiteGuard experiments')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--method', choices=(*METHODS, 'all'), default='all')
    parser.add_argument('--phase', choices=PHASES, default='evaluate')
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parent/'.env', override=False)
    try:
        report = run(load_config(args.config), args.method, args.phase)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    if any(row['status'] in ('blocked', 'failed') for row in report['methods'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
