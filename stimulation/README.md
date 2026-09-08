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

All three datasets expose the same fixed-model and elastic-simulation paths:

| Dataset | Configuration | Features | Classes | Fixed hidden widths | Elastic maximum |
| --- | --- | ---: | ---: | --- | --- |
| CIC-IDS2017 | `configs/cic-ids2017.yaml` | 16 | 2 | 8 / 4 | 16 / 8 |
| UNSW-NB15 | `configs/unsw-nb15.yaml` | 20 | 2 | 32 / 16 | 64 / 32 |
| UNSW-IoT | `configs/unsw-iot.yaml` | 16 | 10 | 64 / 32 | 128 / 64 |

`caravan` always uses `model.class` and the original fixed checkpoint.
Capacity-enabled `drift_conditioned` runs use `model.simulation_class`, embed
that same checkpoint into the active prefix, and load the dataset-specific
`simulation_adapter_checkpoint`. Dormant neurons begin masked and do not alter
the original model output.

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
  and a validated learned adapter to repair selected model parameters. Its
  fixed-capacity repair path has no runtime labeling or gradient retraining;
  the optional software-only capacity experiment described below explicitly
  uses simulation-oracle labels for staged recovery evaluation and selective
  capacity training.

The old `baseline_caravan` spelling is accepted as a deprecated alias for
`caravan`.

`drift_conditioned` adds a teacher-free runtime path:

```text
feature-only histogram/JSD drift
  -> 5-value signature per feature
  -> small drift adapter
  -> predicted hidden modules, magnitudes, feature mask, and confidence
  -> exact-mask-compatible oracle-repair prototypes
  -> replace inactive inputs with frozen reference means
  -> matching small final-classifier repair
  -> repair a copied candidate model
  -> deploy only above the configured confidence threshold
```

Each feature contributes JSD, normalized mean shift, normalized standard-
deviation shift, normalized median shift, and mean-shift direction. Runtime
signature generation accepts feature and histogram state only; it has no label
input and does not construct or call the teacher. The adapter is not a full-weight
hypernetwork. Its added feature head chooses among feature configurations that
have compatible offline repairs, never changes the MLP input width, and never
exceeds the configured active-feature budget. Inactive values are replaced by
the processed-space mean of the frozen reference windows stored in the adapter
checkpoint. The adapter scales the closest same-mask parameter-delta prototype
for each selected module. The repair also carries the same-scenario,
same-mask final-classifier delta.

Ground-truth labels produce oracle repair targets in the offline workflow and
calculate simulation evaluation metrics. They are never part of the drift
signature or adapter input. With capacity disabled, runtime
`drift_conditioned` never performs gradient retraining. With software capacity
enabled, labels are used after weight repair to evaluate residual recovery, to
accept or roll back the feature-only fallback, and for the explicitly logged
capacity train/validation split. Every such use is reported as
`ground_truth_simulation_oracle`; this is not a production label-availability
assumption.

Repair deltas are defined relative to the checkpoint used to generate the
offline dataset. Runtime therefore rebuilds each candidate from that checkpoint
and resolves selected parameters as `checkpoint + delta`, rather than retaining
stale repairs or accumulating deltas across drift episodes. Adapter
checkpoints include and validate the exact model fingerprint,
feature order, histogram width/threshold/reference length/persistence,
signature clip, parameter schema, default mask, feature budget, masking
reference values, prototype masks, and scientific-validation status before use.

### Offline drift-conditioned workflow

Generate controlled single-feature and multi-feature scenarios. Lower-severity
single-feature scenarios form the training split; the highest severity and all
feature combinations are held out for scientific evaluation. Feature pairs are
selected with balanced feature coverage. Oracle optimization uses the first
post-reference traffic window, while every oracle and predicted repair is
scored on the next disjoint traffic window. Each scenario first produces the
existing default-mask weight-only oracle. A deterministic, drift-informed set
of at most `max_candidate_masks` then tests feature masks with a fresh copy of
the same weight oracle. The feature target remains the default unless the best
feature+weight result improves macro F1 by at least `minimum_oracle_gain`.

```bash
cd stimulation
.venv/bin/python scripts/generate_drift_dataset.py \
  --config configs/cic-ids2017.yaml \
  --output results/cic-ids2017-capacity-drift-repair.pt \
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
  --dataset results/cic-ids2017-capacity-drift-repair.pt \
  --output checkpoint/cic-ids2017-capacity-drift-adapter.pt \
  --report results/cic-ids2017-capacity-drift-adapter.report.json \
  --minimum-similarity-correlation 0.10
```

Repeat both commands for each UNSW dataset, changing the configuration and
artifact prefix together. For example, set `DATASET=unsw-nb15` or
`DATASET=unsw-iot` and run:

```bash
.venv/bin/python scripts/generate_drift_dataset.py \
  --config configs/${DATASET}.yaml \
  --output results/${DATASET}-capacity-drift-repair.pt \
  --severities 0.5,1.0,2.0 \
  --directions=-1,1 \
  --max-feature-pairs 16 \
  --oracle-epochs 10 \
  --module-threshold 0.80
.venv/bin/python scripts/train_drift_adapter.py \
  --config configs/${DATASET}.yaml \
  --dataset results/${DATASET}-capacity-drift-repair.pt \
  --output checkpoint/${DATASET}-capacity-drift-adapter.pt \
  --report results/${DATASET}-capacity-drift-adapter.report.json
```

Each adapter is model- and dataset-specific; one dataset's generated adapter
cannot be reused by another.

The report contains drift-signature similarity, oracle-repair similarity,
feature gain/reconfiguration rate, module and feature-mask prediction accuracy,
recovery ratio, and separate held-out results for unseen severities and feature
combinations. Meaningful
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

The elastic CIC artifact measured a pooled held-out similarity correlation of
`0.1055556`, so it did **not** pass the historical default `0.30` correlation
gate. The exact research command above declares a relaxed `0.10` gate; its
other held-out measurements were module accuracy `0.8072917`, recovery ratio
`0.6505885`, oracle-improvement rate `1.0`, and repair coverage `1.0`. This
artifact is suitable only for the disclosed software experiment and is not a
claim that the older gate or hardware-deployment criteria passed. Generated
datasets, reports, and checkpoints are intentionally gitignored and must be
regenerated with these commands in a clean checkout.

That historical adapter report evaluates the original joint mask-compatible
weight prediction. The capacity experiment below reuses its stored current-mask
weight prototypes and learned alternate-mask ranking, but validates the new
strict staged orchestration with a separate controlled runtime trace. It does
not claim a new stage-specific adapter-validation gate.

After producing the checkpoint, run the new policy:

```bash
.venv/bin/python run.py \
  --config configs/cic-ids2017.yaml \
  --mode drift_conditioned
.venv/bin/python run.py \
  --config configs/unsw-nb15.yaml \
  --mode drift_conditioned
.venv/bin/python run.py \
  --config configs/unsw-iot.yaml \
  --mode drift_conditioned
```

Its optional configuration does not change the existing default policy:

```yaml
drift_conditioned:
  adapter_checkpoint: ../checkpoint/cic-ids2017-drift-adapter.pt
  simulation_adapter_checkpoint: ../checkpoint/cic-ids2017-capacity-drift-adapter.pt
  confidence_threshold: 0.50
  module_threshold: 0.50
  signature_shift_clip: 8.0

feature_reconfiguration:
  enabled: true
  active_feature_budget: 16
  minimum_oracle_gain: 0.02
  max_candidate_masks: 32

capacity:
  enabled: true
  initial_layer1: 8
  initial_layer2: 4
  max_layer1: 16
  max_layer2: 8
  residual_f1_threshold: 0.05
  saturation_threshold: 0.75
  layer1_expand_step: 2
  layer2_expand_step: 1
  min_improvement: 0.0
  contraction_enabled: true
  contraction_utility_threshold: 0.01
  contraction_patience: 10
  initialization_noise: 0.001
```

Each drift-conditioned software pipeline uses the dataset-specific masked
supernetwork listed above. Its initial active subnetwork exactly matches the
legacy checkpoint, which is embedded into the leading rows and columns; all
dormant activations are zero. The hardware-facing `model.class` and
`adapter_checkpoint` remain fixed-model artifacts, while `simulation_class`
and `simulation_adapter_checkpoint` isolate elastic software execution. The
default CARAVAN run continues to instantiate the fixed model for every dataset.

The software-capacity policy uses one strict fallback order:

```text
current-mask selective weight repair
  -> evaluate residual recovery
  -> if insufficient, try feature-mask reconfiguration only
  -> accept the mask only if it improves the weight-repaired model; otherwise roll back
  -> if recovery is still insufficient, evaluate neural saturation
  -> capacity reconfiguration
```

Capacity is asserted to be unreachable until a finite, confidence-qualified
weight repair has run and the feature stage is resolved as applied, rejected,
or not applicable. If weight repair initially recovers but deteriorates on a
later persistent-drift window, the pending feature stage still runs before the
first capacity attempt. Pressure then requires both residual F1 loss above the
configured threshold and active-neuron saturation above its threshold. A
deterministic drift-associated subset is split into training data and a
disjoint held-out tail. Expansion training uses `window.labels` as the logged
simulation oracle. The expanded model is accepted only when held-out macro F1
improves by more than `min_improvement`; rejected candidates never replace the
active model. Stable-window contraction uses consecutive low
mean-absolute-activation observations and never crosses the configured initial
widths shown in the table.

`default_feature_mask` may optionally provide one binary value per already
configured dataset feature; otherwise all configured features are active. The
budget must cover that default mask. Candidate generation can disable up to two
strongly drifted active features and, when the default has inactive features,
also considers bounded activations and equal-cardinality swaps. It never creates
new feature computations. Dataset and adapter formats changed for this feature,
so older drift-repair artifacts and adapter checkpoints must be regenerated.

Runtime rows retain the existing metrics and add feature-reconfiguration fields,
per-layer active counts and ratios, residual/saturation measurements, the
simulation label-source disclosure, disjoint capacity train/validation sample
indices, cumulative `repair_stage_order`, separate weight/feature F1 values,
feature acceptance status, and a fixed-schema `capacity_event` record. The JSON
summary also collects expansion, rejected-expansion, and contraction counts and
events.

Confidence below the threshold leaves the model unchanged; it does not invoke
feature or capacity changes as an implicit fallback. Offline oracle repair
training is confined to dataset generation; simulation-oracle capacity training
can run only after weight repair and feature-stage resolution.

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
