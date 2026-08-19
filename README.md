# DriftAdapt

DriftAdapt contains a configuration-driven network-drift emulation pipeline
and an Alveo U55C hardware testbed implementation.

- [`stimulation/`](stimulation/README.md) provides the emulation, labeling,
  retraining, and testbed workload-export pipeline.
- [`testbed/`](testbed/README.md) provides the U55C RTL, OpenNIC integration,
  hardware build and bring-up scripts, packet sender/receiver, and adaptive
  control plane.

Local environments, credentials, downloaded checkpoints, generated packet
workloads, Vivado output, reports, checkpoints, and bitstreams are excluded
from version control. Use the component READMEs and `.env.example` files to
prepare a local deployment.
