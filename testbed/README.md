# DriftAdapt U55C testbed

The testbed exposes exactly two runtime methods:

| Method | Runtime decision | Model update |
|---|---|---|
| `caravan` | Teacher labels produce an accuracy proxy, which triggers online retraining | Full 182-parameter model |
| `drift_conditioned` | Feature histograms produce the validated simulation drift signature, which is evaluated by a pretrained adapter | Only parameter words that differ from the active FPGA image |

`drift_conditioned` is label-free at runtime. It does not construct a teacher,
request labels, read ground truth, form a training set, run an optimizer, or call
online retraining. The drift adapter is trained and scientifically validated by
the `stimulation/` workflow before the hardware run; the testbed only loads it.

## Runtime architecture

The two methods share the same fixed-point student model and feature histogram
monitor:

```text
CARAVAN
FPGA traffic/inference
  -> prediction window
  -> controller teacher labeling
  -> accuracy-proxy trigger
  -> full controller-side retraining
  -> full-model FPGA commit

DriftAdapt (`drift_conditioned`)
FPGA traffic/inference
  -> bounded feature histograms
  -> controller drift signature
  -> validated pretrained adapter
  -> base-relative software repair
  -> changed-index FPGA commit
```

The feature-only path receives the captured Q8.8 feature window together with
reference/current histogram counts, frozen ranges, and window metadata. The
features reproduce and validate the exact simulation signature input; it
receives no label. After reading a `drift_conditioned` window, the controller
uses the ABI v4.2 release operation to free the FPGA bank without submitting
labels or updating proxy confusion counters.

## Offline prerequisite

Create the adapter with the simulation workflow, not in `testbed/`:

```bash
cd /home/zealot/DriftAdapt/stimulation

.venv/bin/python scripts/generate_drift_dataset.py \
  --config configs/cic-ids2017.yaml \
  --output results/cic-ids2017-drift-repair.pt

.venv/bin/python scripts/train_drift_adapter.py \
  --config configs/cic-ids2017.yaml \
  --dataset results/cic-ids2017-drift-repair.pt \
  --output checkpoint/cic-ids2017-drift-adapter.pt
```

The testbed loader rejects an artifact unless its validated confidence and
module thresholds, base-model fingerprint, feature ordering, signature fields
and version, detector/signature configuration, module layout, and prototype
dimensions match the configured runtime.

## Generate assets and build

Regenerate the feature-only traffic ROM and base-model image whenever the
preprocessing or base checkpoint changes:

```bash
cd /home/zealot/DriftAdapt/testbed
../stimulation/.venv/bin/python hardware/u55c/scripts/generate_fpga_assets.py
bash hardware/u55c/scripts/build_u55c.sh all
```

The validated image is written to:

```text
hardware/u55c/build/closure/driftadapt_u55c_timing_closed.bit
```

The sample ROM contains sixteen Q8.8 features per sample and no ground-truth
label. Dataset labels are used only to create the offline manifest reference.

## Program the U55C

Detach the OpenNIC module before replacing the FPGA image:

```bash
sudo modprobe -r onic

cd /home/zealot/DriftAdapt/testbed
bash hardware/u55c/scripts/program_u55c.sh --dry-run
bash hardware/u55c/scripts/program_u55c.sh
```

Do not run two JTAG controller commands concurrently. They share one private
JTAG AXI master.

## Run CARAVAN

CARAVAN preserves the teacher/accuracy-proxy baseline and always deploys a full
model after an accepted retraining event:

```bash
cd /home/zealot/DriftAdapt/testbed
../stimulation/.venv/bin/python -u \
  hardware/u55c/scripts/run_online_adaptation.py \
  --config ../stimulation/configs/cic-ids2017.yaml \
  --mode caravan \
  --provider config \
  --output results/u55c-caravan.jsonl
```

The primary command preserves the labeler type in YAML. Use `--provider dnn`
or `--provider local` only for an explicit CARAVAN labeler override. Provider
selection applies only to CARAVAN.

## Run DriftAdapt

Reprogram the base image before a separate comparison run, then select the
label-free method:

```bash
cd /home/zealot/DriftAdapt/testbed
bash hardware/u55c/scripts/program_u55c.sh --yes

../stimulation/.venv/bin/python -u \
  hardware/u55c/scripts/run_online_adaptation.py \
  --config ../stimulation/configs/cic-ids2017.yaml \
  --mode drift_conditioned \
  --output results/u55c-drift-conditioned.jsonl
```

The adapter path and validated thresholds come from the YAML
`drift_conditioned` section. A low-confidence prediction abstains: the active
model and model version remain unchanged, and no hardware parameter update is
issued.

Every accepted repair is constructed as:

```text
software candidate = validated base model + predicted repair delta
```

It is never computed as `previous active model + delta`. The host quantizes the
complete candidate, compares it with the active FPGA image, and transfers only
the changed `(parameter index, Q16.16 value)` entries. This comparison also
transfers base values needed to remove a repair from an earlier drift episode.
Each window record exposes the predicted modules, exact changed indices,
parameter format, changed count and transfer bytes; parameter values remain in
the private controller-to-FPGA transaction rather than being copied into the
results file.

## Results and latency

Both methods emit a compatible per-window JSONL schema containing the method,
drift/degradation state, adaptation event, model version, changed parameter
count and percentage, update bytes, controller latency, hardware update
latency, and end-to-end recovery latency.

The synthesized U55C sample ROM is deliberately label-free. Consequently,
`drift_conditioned` reports `model_f1_before_drift`, `f1_after_drift`,
`f1_after_predicted_repair`, and `recovery_ratio` as `null`; it never
fabricates task F1 from predictions or feature drift. CARAVAN's online value is
explicitly a teacher-based proxy F1, not ground-truth test-set F1. A separate
offline evaluator may calculate task F1 after a run, but those labels must
remain outside the controller and cannot affect runtime decisions.

CARAVAN additionally records teacher samples, labeling latency, retraining
latency, and full-model update cost. DriftAdapt records drift-detection,
signature, adapter-inference and repair-construction latency, confidence,
predicted modules, and hardware/software equivalence. Its teacher samples,
labeling latency, and online retraining latency are always zero.

The recovery-latency boundaries are:

```text
CARAVAN = label collection + teacher labeling + retraining
          + full-model transfer + hardware commit

DriftAdapt = persistent drift detection + signature construction
             + adapter inference + repair construction
             + changed-parameter transfer + hardware commit
```

Post-adaptation predictions and Q16.16 margins are evaluated against the exact
quantized software candidate. The result reports prediction agreement, exact
margin agreement, parameter mismatches, transferred parameter count, and
transferred bytes. A hardware/software mismatch fails the run rather than being
reported as a valid DriftAdapt result.

An update committed at the end of a truncated stream may have no following
window on which to check deployed outputs. Such a summary is marked
`valid: false` and lists the unverified version in `pending_model_versions`; it
must not be used as an equivalence-validated result.

After the controller stops, inspect the FPGA counters with:

```bash
python3 hardware/u55c/scripts/read_fpga_results.py --json
```
