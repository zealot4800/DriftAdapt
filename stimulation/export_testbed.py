#!/usr/bin/env python3
"""Export a stimulation dataset as the FPGA sender's 16-feature CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

if __package__:
    from .src.data import StreamingDataset
else:
    from src.data import StreamingDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dataset = StreamingDataset(config["dataset"], config_path.parent, torch.device("cpu"))
    if dataset.features.shape[1] != 16:
        parser.error(
            f"the FPGA sender requires 16 features, found {dataset.features.shape[1]}"
        )
    if int(config["dataset"]["num_classes"]) != 2:
        parser.error("the current FPGA dataplane supports only binary classification")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack((dataset.features.numpy(), dataset.labels.numpy()))
    np.savetxt(output, rows, delimiter=",", fmt="%.9g")
    print(f"Exported {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
