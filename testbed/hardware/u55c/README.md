# DRIFTADAPT online-window U55C plugin

## Architecture

```text
driftadapt_packet_generator
  -> driftadapt_dnn_axis (active/shadow Q16.16 weight banks)
  -> driftadapt_window_manager (two 100-sample banks)
       <-> driftadapt_jtag_axi <-> local labeling/retraining agent
  -> driftadapt_benchmark_metrics
```

QDMA and CMAC packet paths remain disabled in this reference implementation.
The private JTAG AXI path transports control-plane windows, returned labels,
and occasional model updates; it is not part of packet inference.

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
| `0x100` | W | submit label bank: bit 0 A, bit 1 B |
| `0x108` | W | commit a complete shadow model |
| `0x114` | R/W | expected window ID for the next label-bank submission |
| `0x1000/0x1100` | R/W | bank A label/valid bitmaps |
| `0x1200/0x1300` | R/W | bank B label/valid bitmaps |
| `0x1400/0x1500` | R | bank A/B prediction bitmaps |
| `0x4000` | R/W | 182 shadow Q16.16 parameters |
| `0x10000/0x20000` | R | bank A/B records, 64-byte stride |

Each record exposes eight feature words followed by sample ID, prediction, and
Q16.16 logit margin. Explicit identifiers prevent stale or reordered agent
responses from being compared with the wrong prediction window.

## Commands

```bash
cd /home/zealot/DriftAdapt/testbed
bash hardware/u55c/scripts/build_u55c.sh all
sudo modprobe -r onic
bash hardware/u55c/scripts/program_u55c.sh
../stimulation/.venv/bin/python -u \
  hardware/u55c/scripts/run_online_adaptation.py
python3 hardware/u55c/scripts/read_fpga_results.py
```
