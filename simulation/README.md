# Packet simulation

CARAVAN, SiteGuard and DriftAdapt share one pipeline:

`packet CSV → chronological time windows → statistics + TLS + DFT/autoencoder → source standardization → classifier → online adaptation → evaluation`

All methods receive identical vectors and predict once per detection window. The shared encoder uses SiteGuard's packet representation: first K data-phase packets per bidirectional session, sampled sessions, 2D DFT amplitudes, a residual autoencoder, 80 traffic statistics and eight TLS fields. Only benign source windows train the autoencoder and DFT ranges. Standardization uses source windows only; the encoder remains fixed during evaluation.

- **CARAVAN** trains a small student and a larger teacher on labeled source windows. It uses teacher agreement as a retraining trigger and balances teacher-labeled examples from the most recent `stream.buffer_windows` windows. The default teacher is a DNN; existing Ollama/OpenAI teachers remain available.
- **SiteGuard** fits its linear one-class detector using benign source windows only.
- **DriftAdapt** keeps the 64-bin evidence encoder and 16–8–2 MLP. After source training and offline threshold calibration it freezes the MLP, threshold and bin boundaries. Online adaptation selectively repairs evidence entries using budgeted paired measurements and fresh reference-relative validation. It uses no target attack labels, source transportation, MLP fine-tuning or threshold updates.

## Input

Supply a timestamp-sorted packet metadata CSV. Existing flow CSVs do not contain the required packet sequences.

Required columns:

```text
timestamp,src_ip,dst_ip,src_port,dst_port,protocol,packet_size,payload_size,tcp_flags,data_phase,label,
tls_has_heartbeat,tls_cipher_suites_cnt,tls_extensions_cnt,tls_has_SNI,tls_sup_heartbeat,
cert_self_signed,cert_validity_duration,cert_SNI_in_SAN
```

Timestamps, session timeout and window width are in seconds. Protocol and TCP flags are numeric; packet/payload lengths are bytes. `data_phase` and TLS flags are 0/1; use zero TLS metadata for non-TLS packets. Labels are binary: benign=0, attack=1. A window is malicious if any packet is malicious. Source and offline packet labels are required; test labels may be blank. A test window with unknown labels and no known attack is excluded from metrics. Labels are never passed to online adaptation.

Set `dataset.packet_file`, `timeline.source_end` and `timeline.adapter_end` in a config. Boundaries must align with `features.window_seconds`. Source and offline windows must each contain both classes for DriftAdapt; at least two benign source windows with data-phase packets are needed by the shared encoder. All methods use the same source/offline/test boundaries. Leading and trailing incomplete windows are excluded. Optional `timeline.evaluation_end` sets an explicit aligned end; otherwise the last complete window boundary is used.

## Run

```bash
pip install -r simulation/requirements.txt
python simulation/run.py --config simulation/configs/cic-ids2017.yaml --method all
```

Methods: `caravan`, `siteguard`, `driftadapt`, `all`. Each invocation trains and evaluates in memory. The terminal prints one line per method with window count, macro-F1, FPR, recall, update count and method runtime. Missing metrics appear as `n/a`. Invalid inputs produce a short error. No result directories, logs, predictions or model checkpoints are written. Separate save/load phases have been removed.

## DriftAdapt observation repair

The evaluator owns two views of each packet window: canonical reference features X and operational features Z. DriftAdapt receives only Z, window identifiers and request/release capabilities. Its training view includes source/offline partitions only. Python capability separation prevents accidental data leakage; it is not an adversarial process sandbox. References represent the same detection window, not per-flow packet labels. Their cost is positive and reported as released pairs and exported bytes on the existing single terminal line.

Optional YAML settings (feature indices refer to the shared standardized vector):

```yaml
observation:
  budget: 256
  delay_windows: 0
  max_pending: 64
  availability: 1.0
  interventions:
    - features: [0]
      start: 100
      end: 300
      shift: 0.5
# driftadapt: {policy: recoverability, max_entries: 32, epsilon: 0.1}
```

Interventions support `scale`, `shift`, `noise`, quantization `step`, `clip: [low, high]`, `missing` probability with replacement `value`, and optional `ramp: true`. Repeated intervals model recurrence. They are **synthetic feature interventions after packet encoding**, not validated physical timing, sampling or packet-length faults. Population/prior changes must be represented in the canonical trace so both views change together. `reference_noise` is an explicit trusted-channel assumption-violation stress test. Fault parameters, onset times and test labels stay evaluator-only.

Requests cost tokens even when duplicated, unavailable or denied by the pending limit; released vectors cost 2d scalar measurements and 16d logical export bytes (float64 pair). Pending and unsuccessful requests remain counted. Unrequested target reference vectors are never released. Fit sampling supports `uniform`, margin-sensitive `decision`, and support/margin-based `recoverability` policies under the same configured token limit. The latter is a cheap uncertainty proxy, not an optimal acquisition policy. Validation always requests the complete, predetermined future cohort; targeted fit sampling cannot select validation cases.

For observed/reference bin pairs, the controller records displacement, fitted scale, noise MAD, sample/bin support, ambiguous mappings and frozen-network sensitivity. A supported unique bin correspondence can propose a boot-LUT value at the corresponding operational bin. Identity and ambiguous regions are preserved. Existing repairs stay untouched outside newly supported regions; supported identity correspondence can propose restoration. Greedy search trades decision disagreement, clipped reference-margin error, changed entries and magnitude. Both total boot-relative changes and current-to-candidate logical word changes are bounded by `max_entries`; no global sparsity optimum is claimed. All candidate decisions use the same floating-point inference path as evaluation. NN parameters, threshold, table shape and finite entries are verified before replacement. The current packet model has no 385-word integer image; this implementation makes no ABI, integer-verification or hardware-throughput claim.

For boot decisions B=f_boot(X), the population optimum for the deployed operational bin vector U is E[min(P(B=0|U),P(B=1|U))]. The controller reports the empirical contradictory-decision fraction over identical **complete** U vectors. This is a finite-sample diagnostic, not a population lower bound. An individual feature's many-to-one mapping does not prove full-decision information loss; absent collisions or unique sparse observations do not prove recoverability. Unsupported regions retain their existing entries. Semantic drift cannot be identified or repaired from these unlabeled pairs alone.

A candidate is frozen before any validation acquisition. Validation uses nonoverlapping blocks; a block fails when any reference decision differs. Let Y_j be this failure indicator and p_j=E[Y_j|past] for the frozen candidate. With fixed lambda=log(100), c=1-exp(-lambda)=0.99, the exponential process exp(c sum p_j - lambda sum Y_j) is a nonnegative supermartingale because exp(c p)(1-c p) <= 1. Consequently, with error probability at most alpha, the cohort-average conditional block failure probability is at most min(1, (lambda sum Y_j + log(1/alpha))/(c n)). This bound allows temporal dependence; grouping windows alone does not create independence. A window's conditional reference-disagreement risk within its block is no greater than that block's failure risk.

Attempt k spends alpha_k=confidence/[k(k+1)], whose sum is confidence. Cohort size is chosen before acquisition and grows when necessary to admit zero failures; the default minimum is 48 blocks of four windows, with epsilon=0.1 and confidence=0.05. Every requested cohort block must be complete. Small budgets, unavailable references, end-of-stream pending pairs or insufficient support retain the current model. Replacement also requires strictly lower empirical disagreement than the current table on that cohort. Restoring boot entries follows the same validation process. No validation cohort fits its candidate, and incomplete/failed validation never silently relaxes epsilon.

For the same classification target and trustworthy paired references, the pointwise indicator inequality implies R(candidate on Z) <= R(boot on X) + P(candidate(Z) != boot(X)). The implemented bound concerns conditional disagreement within the sampled validation cohort; it is **not** an absolute accuracy guarantee or a promise about arbitrary future traffic. Trusted correspondence, reference integrity and equivalent labels remain assumptions. Contract, diagnostic, rejection and telemetry details are returned in memory by `run`; nothing is saved. Historical results and baseline source files are unchanged. The previous staged implementation remains recoverable from Git's index; it is not registered as another method.

## Cleanup validation

The behavior-preserving cleanup was checked against an immutable copy of the pre-cleanup source, using Python 3.11, PyTorch 2.5.1 CPU, one thread and MKLDNN disabled for both versions. The synthetic full-run workload contains 1,920 packets in 240 windows, shared source/offline/test boundaries and two training epochs. Timings are medians of four warm runs; the isolated repair workload uses 32 paired examples and 32 greedy edits, timed seven times.

| Measurement | Before | After |
| --- | ---: | ---: |
| Full simulation | 0.6088 s | 0.5310 s (12.8% less time) |
| Packet preprocessing | 0.4727 s | 0.4083 s (13.6% less time) |
| 888-feature repair search | 0.1553 s | 0.0988 s (36.3% less time) |
| 128-feature repair search | 0.0496 s | 0.0379 s (23.7% less time) |
| Python runtime lines | 1,297 | 1,308 |
| Nonblank, noncomment lines | 1,166 | 1,176 |
| Files / functions / classes | 13 / 80 / 15 | 13 / 81 / 15 |
| Process peak RSS | 391.9 MiB | 394.0 MiB |
| Generated result files | 0 | 0 |

Source counts cover `run.py` and `src/**/*.py`, excluding tests/configuration. Source grew 0.85% (code reduction: -0.85%); the extra normalization helper and explicit per-search reuse remove repeated work without deleting supported behavior. No peak-memory reduction is demonstrated. The redundant direct SciPy requirement was removed; scikit-learn still installs SciPy transitively.

All 39 original tests and one new boundary/empty-window regression pass. Exact comparisons preserve feature/model hashes, predictions, repair candidates, diagnostics, RNG state, metrics and telemetry. Both script and module CLI entry points remain valid. Configurations, defaults, algorithms, validation contracts and baseline implementations are unchanged. Real packet captures, CUDA and external LLM services were not exercised by these benchmarks.

No new methodological bug was identified. The feature-intervention and cohort-bound limitations above remain unchanged. One pre-existing out-of-scope consumer remains incompatible: `testbed/hardware/u55c/scripts/driftadapt_runtime.py` imports removed flow/feedback APIs. That script and all other non-simulation files were left untouched.
