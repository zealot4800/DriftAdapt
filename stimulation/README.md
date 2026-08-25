# DriftAdapt simulation

This directory contains the CARAVAN baseline and the experimental
drift-conditioned DriftAdapt workflow. Both use the same streaming datasets,
MLPs, feature ordering, checkpoints, histogram/JSD drift monitor, and result
formats. The runtime interface intentionally exposes only `caravan` and
`drift_conditioned`.

## Install and run

Use Python 3.10 or newer in a virtual environment:

```bash
python -m pip install -r requirements.txt
python run.py --config configs/cic-ids2017.yaml --mode caravan
```

On a CPU-only host, install the CPU wheel before the requirements to avoid
downloading CUDA libraries:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

CUDA is used when requested and available; otherwise execution falls back to
CPU. Results are written to `results/<dataset>.csv` and
`results/<dataset>.json`. The three consolidated official CSVs live in `data/`, while
pretrained weights live in `checkpoint/`. These files are unchanged copies
or order-preserving concatenations of files from the official repository.

UNSW-NB15 uses a local Llama 2 7B labeler through Ollama. It does not require
an OpenAI account, API key, or Internet access after the model is installed:

```bash
ollama pull llama2:7b
python run.py --config configs/unsw-nb15.yaml --mode caravan --local
```

Local-model settings live in the project-root `.env` file:

```dotenv
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama2:7b
OLLAMA_BATCH_SIZE=10
OLLAMA_TIMEOUT=300
```

`run.py` loads this file automatically. If Ollama is not managed as a system
service, start it with `ollama serve`. The quantized 7B model runs on CPU when
a supported GPU is unavailable. `OLLAMA_BATCH_SIZE` limits prompts so they fit
the model's context window. `.env.example` is the commit-safe template.

### OpenAI provider

The same workload can instead use OpenAI's Responses API. Put the key only in
the ignored `.env` file and choose a model available to your API project:

```dotenv
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.4-nano
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_BATCH_SIZE=10
OPENAI_TIMEOUT=300
OPENAI_MAX_RETRIES=3
```

Then select the cloud provider with the same configuration:

```bash
python run.py --config configs/unsw-nb15.yaml --mode caravan --cloud
```

The `--local` and `--cloud` flags are mutually exclusive; the shorter aliases
`-local` and `-cloud` are also accepted. If neither is supplied, the YAML's
`labeler.type` is used (Ollama by default). Both providers use the same prompt,
feature order, strict binary-label schema, batching, validation, and DRIFTADAPT
training pipeline. Explicit provider runs receive `-local` or `-cloud` result
name suffixes so they do not overwrite each other.

## Configuration

Paths are resolved relative to the YAML file. Multiple files may be listed in
`dataset.files`; they are independently standardized exactly as in the
artifact and then concatenated in listed temporal order. This is a drift-only
implementation: every configuration retains the `accuracy_proxy` trigger for
the CARAVAN baseline. There is no continuous or fixed-frequency retraining
path.

The independent `drift` configuration controls a label-free feature monitor:
`bins` fixes the histogram width, `reference_windows` learns and freezes the
per-feature ranges, and `threshold` plus `consecutive_windows` determine when a
feature is reported as drifted. This monitor observes only input features.
Results include per-feature JSD scores, drifted indices and names, and the
fixed histogram counter/state cost for later hardware sizing.

`driftadapt.mode` selects one of two runtime policies:

- `caravan` labels every complete window, applies the original accuracy-proxy
  trigger, fully retrains the active model, and preserves the CARAVAN baseline.
- `drift_conditioned` uses persistent feature drift, a feature-only signature,
  and a validated learned adapter to repair selected model parameters without
  runtime labeling or gradient retraining.

The old `baseline_caravan` spelling is accepted as a deprecated alias for
`caravan`.

`drift_conditioned` adds a teacher-free runtime path:

```text
feature-only histogram/JSD drift
  -> 5-value signature per feature
  -> small drift adapter
  -> predicted hidden modules, magnitudes, and confidence
  -> nearest learned oracle-repair prototype per predicted module
  -> matching small final-classifier repair
  -> repair a copied candidate model
  -> deploy only above the configured confidence threshold
```

Each feature contributes JSD, normalized mean shift, normalized standard-
deviation shift, normalized median shift, and mean-shift direction. Runtime
signature generation accepts feature and histogram state only; it has no label
input and does not construct or call the teacher. The adapter is not a full-weight
hypernetwork. It predicts hidden-module selection and normalized magnitude and
scales the closest offline parameter-delta prototype for each selected module.
The repair also carries the matching final-classifier delta. This is required
because a hidden-only repair cannot reproduce an oracle whose classifier was
adapted, while keeping the adapter output module-oriented and explicit.

Ground-truth labels produce oracle repair targets in the offline workflow and
calculate simulation evaluation metrics. They are never part of the drift
signature or adapter input, and runtime `drift_conditioned` never performs
gradient retraining. Oracle retraining is confined to offline artifact
generation and evaluation.

Repair deltas are defined relative to the checkpoint used to generate the
offline dataset. Runtime therefore rebuilds each candidate from that checkpoint
and resolves selected parameters as `checkpoint + delta`, rather than retaining
stale repairs or accumulating deltas across drift episodes. Adapter
checkpoints include and validate the exact model fingerprint,
feature order, histogram width/reference length, signature clip, parameter
schema, and scientific-validation status before use.

### Offline drift-conditioned workflow

Generate controlled single-feature and multi-feature scenarios. Lower-severity
single-feature scenarios form the training split; the highest severity and all
feature combinations are held out for scientific evaluation. Feature pairs are
selected with balanced feature coverage. Oracle optimization uses the first
post-reference traffic window, while every oracle and predicted repair is
scored on the next disjoint traffic window.

```bash
cd stimulation
.venv/bin/python scripts/generate_drift_dataset.py \
  --config configs/cic-ids2017.yaml \
  --output results/cic-ids2017-drift-repair.pt \
  --severities 0.5,1.0,2.0 \
  --directions=-1,1 \
  --max-feature-pairs 16 \
  --oracle-epochs 10 \
  --module-threshold 0.80
```

Train the adapter and produce the learnability/generalization report:

```bash
.venv/bin/python scripts/train_drift_adapter.py \
  --config configs/cic-ids2017.yaml \
  --dataset results/cic-ids2017-drift-repair.pt \
  --output checkpoint/cic-ids2017-drift-adapter.pt \
  --report results/cic-ids2017-drift-adapter.report.json
```

The report contains drift-signature similarity, oracle-repair similarity,
their correlation, module-prediction accuracy, recovery ratio, and separate
held-out results for unseen severities and feature combinations. Meaningful
held-out recovery should be demonstrated before changing the hardware testbed.
The report's `scientific_validation_passed` gate defaults to requiring at least
`0.30` pooled held-out signature/repair similarity correlation, `0.70`
held-out module accuracy, `0.50` held-out recovery ratio, and a `0.50` rate of
held-out scenarios where oracle adaptation actually improves F1. Scenarios
without positive oracle gain are reported but excluded from recovery-ratio
averages. These thresholds are explicit CLI options rather than an implicit
claim of success. The script always writes the report, but writes a deployable
adapter checkpoint only when every gate passes.
The evaluated confidence and module thresholds are stored in the checkpoint;
changing either runtime threshold requires regenerating the validated adapter.

After producing the checkpoint, run the new policy:

```bash
.venv/bin/python run.py \
  --config configs/cic-ids2017.yaml \
  --mode drift_conditioned
```

Its optional configuration does not change the existing default policy:

```yaml
drift_conditioned:
  adapter_checkpoint: ../checkpoint/cic-ids2017-drift-adapter.pt
  confidence_threshold: 0.50
  module_threshold: 0.50
  signature_shift_clip: 8.0
```

Confidence below the threshold leaves the model unchanged; it does not invoke
the teacher or gradient retraining as an implicit fallback. Oracle retraining
is confined to the offline dataset-generation command above.

## CIC-IDS2017 traffic-drift run

`configs/cic-ids2017.yaml` reads the consolidated `data/cic-ids2017.csv`, which
contains workload 1 followed by workload 2. With 7,000 workload-1 flows and a
100-flow window, workload 2 begins at window 71. DRIFTADAPT does not receive that
boundary during execution; it evaluates every window normally. Configuration
segment lengths preserve the artifact's original per-workload standardization.

```bash
python run.py --config configs/cic-ids2017.yaml --mode caravan
python run.py --config configs/cic-ids2017.yaml --mode drift_conditioned
```

The optional CLI flag overrides `driftadapt.mode`; without it, the YAML value
is used. `--mode baseline_caravan` remains temporarily accepted as a deprecated
CARAVAN alias.

To regenerate the feature and model memories consumed by the U55C image:

```bash
../stimulation/.venv/bin/python \
  ../testbed/hardware/u55c/scripts/generate_fpga_assets.py
```

This configuration uses the DNN labeling agent to calculate the student's
accuracy-proxy F1. The minimum proxy is `0.20`, the abrupt-drop threshold is
`0.15`, and two consecutive low-proxy windows are required. Ground-truth labels
are used only for simulation reporting and offline asset verification; they are
not used by the trigger, hardware window manager, or online training.

## Provenance and scope

Reference: [Per-Packet-AI/DriftAdapt-Artifact-OSDI24](https://github.com/Per-Packet-AI/DriftAdapt-Artifact-OSDI24).
The implementation deliberately excludes rule caching, plotting, P4 programs,
notebooks, artifact figure drivers, and hardware-aware execution. The U55C
integration is documented in `../testbed/README.md`.
