# Minimal DRIFTADAPT baseline

This directory is a compact implementation of the DRIFTADAPT online-learning
pipeline from the official OSDI '24 artifact. It contains only streaming data
loading, the artifact's required MLPs and labelers, retraining triggers,
full-model training, metrics, official workloads, and official checkpoints.

## Install and run

Use Python 3.10 or newer in a virtual environment:

```bash
python -m pip install -r requirements.txt
python run.py --config configs/cic-ids2017.yaml
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
python run.py --config configs/unsw-nb15.yaml --local
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
python run.py --config configs/unsw-nb15.yaml --cloud
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
implementation: every configuration requires the `accuracy_proxy` trigger.
There is no continuous or fixed-frequency retraining path.

For each window, DRIFTADAPT compares the current student's predictions with labels
from its labeling agent. It fully retrains only when that proxy F1 is below the
configured threshold and the labeled training set is usable. Binary workloads
use the artifact's random undersampling to form a balanced online set. Every
parameter remains trainable. Multiclass IoT workloads use all device-list-
matched samples without binary balancing.

## CIC-IDS2017 traffic-drift run

`configs/cic-ids2017.yaml` reads the consolidated `data/cic-ids2017.csv`, which
contains workload 1 followed by workload 2. With 7,000 workload-1 flows and a
100-flow window, workload 2 begins at window 71. DRIFTADAPT does not receive that
boundary during execution; it evaluates every window normally. Configuration
segment lengths preserve the artifact's original per-workload standardization.

```bash
python run.py --config configs/cic-ids2017.yaml
```

To regenerate the packet workload consumed by the FPGA sender from this exact
feature order and per-segment preprocessing:

```bash
python export_testbed.py --config configs/cic-ids2017.yaml \
  --output ../testbed/sendrecv/data/intrusion-detection.csv
```

This configuration uses the DNN labeling agent to calculate the student's
accuracy-proxy F1. Retraining is requested only when proxy F1 is below `0.20`.
That cutoff was selected from the unchanged pretrained model: 0 of 70 windows
in workload 1 and 18 of 90 windows in workload 2 fall below it before online
adaptation. Ground-truth labels are used only for reporting `student_f1` and
are not used by the trigger or DRIFTADAPT training.

## Provenance and scope

Reference: [Per-Packet-AI/DriftAdapt-Artifact-OSDI24](https://github.com/Per-Packet-AI/DriftAdapt-Artifact-OSDI24).
The implementation deliberately excludes rule caching, plotting, P4/FPGA
code, notebooks, artifact figure drivers, and any selective or partial
retraining extensions.
