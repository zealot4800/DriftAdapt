# DriftAdapt

DriftAdapt contains a configuration-driven network-drift emulation pipeline
and an Alveo U55C hardware testbed implementation.

- [`stimulation/`](stimulation/README.md) provides the emulation, labeling,
  retraining, and FPGA asset-generation source data.
- [`testbed/`](testbed/README.md) provides the U55C data-plane DNN, dual
  labeling-window buffers, local-agent JTAG control plane, accuracy-proxy
  validation, atomic model updates, and performance counters.

Local environments, credentials, downloaded checkpoints, generated packet
exports, online result logs, Vivado output, reports, checkpoints, and
bitstreams are excluded from version control. Linux networking and an optical
link are not part of the reference window experiment.
