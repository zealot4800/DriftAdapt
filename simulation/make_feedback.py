#!/usr/bin/env python3
"""Generate a verified-feedback schedule (JSONL) for the evaluation partition.

A seeded random fraction (--budget) of test flows receives its ground-truth label,
released --delay windows after the window in which the flow was predicted.
--flip corrupts that fraction of the released labels (label-quality experiments).
Default output: feedback/<dataset>-b<budget%>-d<delay>-s<seed>.jsonl
"""
import argparse
import json
from pathlib import Path
import numpy as np
from src.config import load_config
from src.data import FlowData

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--budget', type=float, default=0.05, help='fraction of test flows labeled')
    parser.add_argument('--delay', type=int, default=2, help='release delay in windows')
    parser.add_argument('--flip', type=float, default=0.0, help='fraction of released labels made wrong')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not 0 <= args.budget <= 1 or not 0 <= args.flip <= 1 or args.delay < 0:
        parser.error('budget and flip must be in [0, 1]; delay must be nonnegative')

    config = load_config(args.config)
    name, size = config['dataset']['name'], config['stream']['window_size']
    classes = config['dataset']['num_classes']
    labels = FlowData(config).labels[2].cpu().numpy()
    count = len(labels)//size*size  # only complete windows are evaluated

    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(count, round(count*args.budget), replace=False))
    released = labels[chosen].astype(int)
    flipped = rng.random(len(chosen)) < args.flip
    for i in np.flatnonzero(flipped):
        released[i] = int(rng.choice([c for c in range(classes) if c != released[i]]))

    out = args.out or HERE/'feedback'/f'{name}-b{round(args.budget*100):02d}-d{args.delay}-s{args.seed}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as handle:
        for sample, label in zip(chosen.tolist(), released.tolist()):
            handle.write(json.dumps({'sample_id':sample, 'label':label, 'verified':True,
                                     'available_window':sample//size+1+args.delay})+'\n')
    print(f'{out}: {len(chosen)} labels ({int(flipped.sum())} flipped) of {count} flows')


if __name__ == '__main__':
    main()
