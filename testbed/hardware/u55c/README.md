# DriftAdapt U55C implementation

The U55C implementation supports the testbed's two controller methods:

- `caravan`: teacher labels, FPGA accuracy-proxy scoring, controller-side full
  retraining, and a 182-word model commit.
- `drift_conditioned`: feature-only histogram monitoring, controller-side
  pretrained adapter inference, base-relative repair, and a changed-word
  commit.

The learned adapter remains on the controller. The FPGA contains only traffic
generation, fixed-point inference, bounded histogram state, two stable window
banks, model banks, and update accounting.

## Dataplane and control plane

```text
driftadapt_packet_generator
  +-> driftadapt_dnn_axis (active/shadow Q16.16 model banks)
  +-> driftadapt_hist_monitor (feature-only Q8.8 histogram tap)
  +-> driftadapt_window_manager (two stable 100-sample banks)
             |
             +-> CARAVAN: labels -> proxy counters -> full update
             |
             +-> DriftAdapt: label-free release -> localized update
             |
       private JTAG AXI -> host controller
```

QDMA and CMAC packet paths remain disabled in this reference implementation.
The private JTAG AXI path is control-plane transport and is not part of packet
inference.

Each window record contains sixteen features, sample ID, FPGA prediction, and
Q16.16 logit margin. No ground-truth label is stored in the sample ROM or
window record. CARAVAN writes generated label and validity bitmaps before
scoring. DriftAdapt instead issues a window-ID-checked release command; release
does not write label memory, score predictions, or increment labeled counters.

## Histogram monitor

The monitor observes the exact accepted sixteen-feature packet seen by the
DNN. It uses no prediction, teacher output, label, proxy score, or confusion
counter. During the configured reference windows it learns signed Q8.8
minimum/maximum ranges and accumulates reference histograms. Ranges then freeze.
It exposes stable reference/current counts and ranges to the controller, where
base-2 JSD, persistence, the simulation drift signature, and adapter inference
are calculated.

The fixed 16-feature, 16-bin monitor contains:

- 256 reference counters;
- 256 current accumulators;
- two 256-counter stable snapshots;
- sixteen signed minimum/maximum pairs.

The 1,024 saturating 32-bit counters occupy 4,096 bytes; ranges occupy 64
bytes. The reported bounded histogram state is therefore 4,160 bytes.

## Assets

- `assets/driftadapt_samples.mem`: 16,000 feature-only Q8.8 vectors.
- `assets/driftadapt_weights.mem`: base student model, 182 Q16.16 words.
- `assets/manifest.json`: reproducibility hashes and offline fixed-point
  reference results.

Generate them with:

```bash
cd /home/zealot/DriftAdapt/testbed
../stimulation/.venv/bin/python hardware/u55c/scripts/generate_fpga_assets.py
```

## Control ABI v4.2

The private control ABI version is `0x00040002`. Important registers are:

| Offset | Access | Meaning |
|---:|:---:|---|
| `0x000` | R | Feature word `0x44524654` |
| `0x004` | R | Generator/window/label/model status |
| `0x008` | R | ABI `0x00040002` |
| `0x018` | R | Window size |
| `0x020` | R | Ready mask: bit 0 bank A, bit 1 bank B |
| `0x024..0x038` | R | Bank window ID, count, and first sample ID |
| `0x044/0x048` | R | Model version used by bank A/B inference |
| `0x04c` | R | Model version used by the last scored window |
| `0x050..0x068` | R | Last CARAVAN score: window, TP/TN/FP/FN/matches |
| `0x06c` | R | CARAVAN windows labeled |
| `0x070` | R | CARAVAN samples labeled, 64-bit |
| `0x078` | R | Active model version |
| `0x0d0` | R | Closed-loop cycles, 64-bit |
| `0x0d8` | R | Histogram status: snapshot/reference/error bits |
| `0x0dc/0x0e0` | R | Histogram window ID and sample count |
| `0x0e4` | R | Feature/bin/reference-window configuration |
| `0x0e8` | R/W | Histogram feature/bin/bank selector |
| `0x0ec/0x0f0` | R | Selected reference/current count |
| `0x0f4/0x0f8` | R | Selected signed Q8.8 range |
| `0x0fc` | R | Histogram state bytes |
| `0x100` | W | Window command: bit 0/1 submit CARAVAN bank A/B; bit 2/3 release DriftAdapt bank A/B without labels |
| `0x108` | W | Commit shadow model |
| `0x114` | R/W | Expected submit/release window ID |
| `0x118` | R/W | Update control/status |
| `0x11c/0x120` | R | Unique patched parameters / transferred bytes |
| `0x124..0x140` | R | Clone, patch, commit, and total update cycles |
| `0x144/0x148` | R | Old/new model version |
| `0x14c` | R | Active model version |
| `0x150` | R | DriftAdapt windows released |
| `0x154` | R | Last released window ID |
| `0x158` | R | Last released window sample count |
| `0x15c/0x160` | R | Total released samples, 64-bit |
| `0x1000..0x1300` | R/W | CARAVAN label/valid bitmaps |
| `0x1400/0x1500` | R | Prediction bitmaps |
| `0x4000` | R/W | 182 shadow Q16.16 parameters |
| `0x10000/0x20000` | R | Bank A/B records, 64-byte stride |

The v4.2 label-free release operation validates the requested bank and window
ID before freeing it. Stale, reordered, or already-consumed releases fail.

## Full and localized model commits

CARAVAN stages all 182 words in the inactive bank and atomically commits the
complete model. Its parameter payload is:

```text
182 parameters * 4 bytes = 728 bytes
```

For DriftAdapt, the host first constructs and quantizes the complete
base-relative software candidate:

```text
candidate = validated base + predicted delta
```

It compares that candidate with its tracked active FPGA words. Only differing
indices and new signed Q16.16 values are sent. Results retain the module names,
indices, precision, count, and byte total without duplicating model values. On
the FPGA:

```text
active bank vN
  -> clone all words into inactive bank
  -> patch requested indices
  -> verify every unpatched word equals the old active bank
  -> atomically switch banks
  -> active bank vN+1
```

The clone is an internal safety operation; it is not counted as controller
transfer bytes. Transferred bytes are exactly `4 * unique changed indices`.
Duplicate or out-of-range indices, incomplete full updates, verification
failure, and empty localized updates are rejected without advancing the model
version.

Because changed indices are calculated against the active image, the patch may
include base values that remove a previous repair. Consequently the hardware
ends with the new base-relative candidate rather than accumulated deltas.

## Hardware/software equivalence

After an accepted DriftAdapt update, validation packets are evaluated by both:

```text
quantized software candidate <-> deployed FPGA candidate
```

The controller reports prediction agreement, exact Q16.16 margin agreement,
parameter mismatches, transferred parameter count, and transferred bytes. An
equivalence failure invalidates the adaptation event. Low adapter confidence
performs no commit and leaves the version and active model unchanged.

If a truncated stream ends immediately after a commit, there is no subsequent
hardware window for output validation. The summary is then `valid: false` and
lists that version under `pending_model_versions`.

Because the synthesized traffic ROM has no labels, DriftAdapt's online
pre-drift, post-drift and post-adaptation task F1 and recovery ratio are `null`.
Fixed-point prediction/margin agreement is the online correctness check.
CARAVAN reports teacher proxy F1. Ground-truth task F1, if desired, must be
calculated by a separate offline evaluator and cannot feed the controller.

## Build and run

```bash
cd /home/zealot/DriftAdapt/testbed
bash hardware/u55c/scripts/build_u55c.sh all
sudo modprobe -r onic
bash hardware/u55c/scripts/program_u55c.sh --yes

# CARAVAN
../stimulation/.venv/bin/python -u \
  hardware/u55c/scripts/run_online_adaptation.py \
  --config ../stimulation/configs/cic-ids2017.yaml \
  --mode caravan \
  --provider config \
  --output results/u55c-caravan.jsonl

# DriftAdapt
../stimulation/.venv/bin/python -u \
  hardware/u55c/scripts/run_online_adaptation.py \
  --config ../stimulation/configs/cic-ids2017.yaml \
  --mode drift_conditioned \
  --output results/u55c-drift-conditioned.jsonl

python3 hardware/u55c/scripts/read_fpga_results.py --json
```

The adapter checkpoint referenced by the YAML must already exist and pass the
simulation loader's scientific-validation and compatibility checks. No adapter
training is implemented in `testbed/`.
