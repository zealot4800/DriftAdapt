# DRIFTADAPT online-window U55C plugin

## Architecture

```text
driftadapt_packet_generator
  +-> driftadapt_dnn_axis (active/shadow Q16.16 weight banks)
  |    -> driftadapt_window_manager (two 100-sample banks)
  +-> driftadapt_hist_monitor (feature-only Q8.8 histogram tap)
       <-> driftadapt_jtag_axi <-> local labeling/retraining/JSD agent
  -> driftadapt_benchmark_metrics
```

QDMA and CMAC packet paths remain disabled in this reference implementation.
The private JTAG AXI path transports control-plane windows, returned labels,
and occasional model updates; it is not part of packet inference.

The histogram tap observes the same accepted sixteen-feature packet as the DNN.
It uses no prediction, label, teacher output, proxy score, or confusion counter.
During the configured reference windows it learns signed Q8.8 minima/maxima and
accumulates the reference histogram. Ranges then freeze. One feature is binned
per cycle with fixed cross-multiply comparisons, well inside the DNN's 672-cycle
packet interval; no divider, logarithm, floating-point unit, or backpressure is
added. Two completed-window histogram banks track the existing two labeling
windows, so the host always reads a stable matching snapshot.

`driftadapt_window_manager` assigns window and sample IDs and records sixteen
feature-only values, the data-plane prediction, and
the absolute logit margin. It refuses a label submission unless every sample
has a valid agent label. Scoring takes one fabric cycle per sample and frees
the bank only after its proxy confusion matrix is complete.

## Memory assets

- `assets/driftadapt_samples.mem`: 16,000 reproducible feature-only vectors;
  no ground-truth label is synthesized into this ROM.
- `assets/driftadapt_weights.mem`: initial 182-word Q16.16 student model.
- `assets/manifest.json`: source hashes and offline fixed-point reference.

## Control ABI v4.1

Important registers are:

| Offset | Access | Value |
|---:|:---:|---|
| `0x000` | R | feature word `0x44524654` |
| `0x004` | R | generator/window/label/model status bits |
| `0x008` | R | ABI `0x00040001` |
| `0x018` | R | labeling window size |
| `0x020` | R | ready mask: bit 0 bank A, bit 1 bank B |
| `0x024..0x038` | R | bank window ID, count, and first sample ID |
| `0x050..0x068` | R | most recently scored window and TP/TN/FP/FN/matches |
| `0x06c` | R | number of labeled windows |
| `0x070` | R | total labeled samples, 64-bit |
| `0x078` | R | active model version |
| `0x0d0` | R | first-input through final-label closed-loop cycles, 64-bit |
| `0x0d8` | R | histogram status: bit 0 snapshot valid, bit 1 reference ready, bit 3 saturation/protocol error |
| `0x0dc` | R | latest completed histogram-window ID |
| `0x0e0` | R | latest completed histogram-window sample count |
| `0x0e4` | R | histogram configuration: features `[31:24]`, bins `[23:16]`, reference windows `[15:0]` |
| `0x0e8` | R/W | histogram selector: feature `[7:0]`, bin `[15:8]`, snapshot bank `[16]` |
| `0x0ec` | R | selected reference counter |
| `0x0f0` | R | selected completed-window counter |
| `0x0f4/0x0f8` | R | selected feature signed Q8.8 minimum/maximum |
| `0x0fc` | R | histogram storage bytes |
| `0x100` | W | submit label bank: bit 0 A, bit 1 B |
| `0x108` | W | commit a complete shadow model |
| `0x114` | R/W | expected window ID for the next label-bank submission |
| `0x118` | R/W | update control/status; W bit 0 starts selective clone, bit 1 starts full-update accounting, bit 2 clears protocol error |
| `0x11c` | R | unique patched-parameter count |
| `0x120` | R | transferred update bytes |
| `0x124/0x128` | R | active-to-shadow clone cycles, 64-bit |
| `0x12c/0x130` | R | parameter patch/transfer cycles, 64-bit |
| `0x134/0x138` | R | verification and atomic-commit cycles, 64-bit |
| `0x13c/0x140` | R | total hardware-update cycles, 64-bit |
| `0x144/0x148` | R | old/new model version |
| `0x14c` | R | active model version |
| `0x1000/0x1100` | R/W | bank A label/valid bitmaps |
| `0x1200/0x1300` | R/W | bank B label/valid bitmaps |
| `0x1400/0x1500` | R | bank A/B prediction bitmaps |
| `0x4000` | R/W | 182 shadow Q16.16 parameters |
| `0x10000/0x20000` | R | bank A/B records, 64-byte stride |

Each record exposes eight feature words followed by sample ID, prediction, and
Q16.16 logit margin. Explicit identifiers prevent stale or reordered agent
responses from being compared with the wrong prediction window.

## Histogram resources and host detection

The fixed 16-feature, 16-bin implementation contains 1,024 saturating 32-bit
counters: 256 reference counters, 256 current accumulators, and two sets of 256
stable snapshot counters. They require 32,768 bits (4,096 bytes). The 16 signed
minimum/maximum pairs add 512 bits (64 bytes), for 33,280 bits (4,160 bytes) of
reported histogram state. Small selectors, IDs, flags, and sample staging are
control logic and are reported separately by implementation utilization.

`run_online_adaptation.py` reads `drift.bins`, `threshold`,
`consecutive_windows`, and `reference_windows` from the normal stimulation YAML.
It checks the hardware configuration, computes the same base-2 JSD function as
`stimulation/src/drift.py`, applies per-feature persistence, and appends the
feature scores and drift vector to every existing JSONL result row. The selected
host policy decides whether accuracy-proxy or feature drift requests adaptation;
neither trigger is implemented inside the FPGA datapath.

## Full and selective model updates

The original full-model path remains available: stage all 182 Q16.16 words in
the inactive bank and write `0x108` to switch banks at an inference boundary.
Selective updates use the following deterministic sequence:

```text
active bank A (vN)
  -> clone all 182 words into inactive bank B
  -> patch only host-selected indices in B
  -> compare every unpatched B word with active A
  -> atomically switch to B only if verification succeeds
  -> active bank B (vN+1)
```

Inference is blocked only during clone, verification, and the atomic switch. It
may continue from the active bank while JTAG writes selective patches into the
inactive bank. A failed unchanged-word comparison prevents the switch and does
not increment the model version.

Host modes are:

- `caravan`: accuracy-proxy trigger, host full retraining, 182-word transfer.
- `driftadapt-selective`: feature-drift trigger, host impact localization and selective retraining, changed-word transfer only.

## Commands

```bash
cd /home/zealot/DriftAdapt/testbed
bash hardware/u55c/scripts/build_u55c.sh all
sudo modprobe -r onic

# CARAVAN full-model baseline
bash hardware/u55c/scripts/program_u55c.sh --yes
../stimulation/.venv/bin/python -u \
  hardware/u55c/scripts/run_online_adaptation.py \
  --config ../stimulation/configs/cic-ids2017.yaml \
  --provider config \
  --mode caravan \
  --output results/u55c-caravan.jsonl

# DriftAdapt selective FPGA patching
bash hardware/u55c/scripts/program_u55c.sh --yes
../stimulation/.venv/bin/python -u \
  hardware/u55c/scripts/run_online_adaptation.py \
  --config ../stimulation/configs/cic-ids2017.yaml \
  --provider config \
  --mode driftadapt-selective \
  --output results/u55c-driftadapt-selective.jsonl

python3 hardware/u55c/scripts/read_fpga_results.py
```
