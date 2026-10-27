# U55C encoded MLP testbed

The only active hardware architecture is the current simulation DriftAdapt,
`distilled_evidence_quantized_mlp_v4`. FPGA inference uses the checkpoint in
`simulation/results/artifacts/cic-ids2017-driftadapt-mlp-v4.pt`.

## Architecture

Source-only preprocessing and feature binning happen on the host. A replay ROM
supplies sixteen encoder-bin IDs per flow; it contains no labels. On the FPGA:

1. A 16 × 64 signed int8 evidence table encodes the bin IDs.
2. A 16 → 16 → 8 → 2 MLP computes logits using signed int8 weights,
   signed int32 biases, and ReLU/uint8 hidden activations.
3. Hidden activations use integer requantization with 64-bit intermediates and
   a 24-bit right shift. A signed int32 margin is compared with the calibrated
   threshold using `>=`, matching simulation.
4. A frozen source histogram monitors total variation over sixteen bins per
   feature. Three consecutive shifted windows trigger drift with the CIC config.
5. Two stable window buffers hold bin IDs, prediction, signed margin and the
   model version **for every record**.

The host imports the same DriftAdapt implementation as simulation. It retains
the original flow features, applies only causally released verified feedback,
retrains the MLP and recalibrates its threshold. The source encoder stays frozen.
Without a configured feedback file there is no online training.

Updates stage all 385 words in the inactive model bank. A commit checks that
every word was written and swaps banks only between inferences. Reset reloads
both banks from the original ROM. A packet already in flight keeps its original
version. The controller validates every prediction and margin against that
version's integer model image before releasing its window.

There is no in-fabric training, packet reconstruction, teacher-label scoring,
or live QDMA/CMAC traffic path. The shell ports are disconnected as in the
internal replay testbed. Performance and power require a new board measurement;
old bitstreams and reports are incompatible.

## Export and verify

From the repository root, using the simulation environment:

```bash
simulation/.venv/bin/python testbed/hardware/u55c/scripts/generate_fpga_assets.py
simulation/.venv/bin/python testbed/hardware/u55c/scripts/verify_rtl.py
simulation/.venv/bin/python testbed/hardware/u55c/scripts/run_online_adaptation.py \
  --backend software --output testbed/results/encoded-mlp-software.json
```

The exporter uses the real source/offline/evaluation partitions and checks that
FPGA monitor bins align with simulation bins. It exports complete evaluation
windows only (13,800 CIC flows with the current configuration). Checksums and a
build ID bind the model, reference, traffic ROM and generated constants.
Software replay explicitly reports `hardware_verified: false`.

RTL verification uses Vivado 2022.2 XSim. It checks all replay margins, prediction
decisions, histogram triggers, window contents, backpressure, invalid releases,
incomplete updates, atomic threshold changes and reset restoration. Its
artifacts live under `build/encoded_mlp_v5/verification/`.

Host protocol and feedback regression checks:

```bash
simulation/.venv/bin/python -m unittest discover \
  -s testbed/hardware/u55c/scripts -p 'test_*.py' -v
```

The standalone synthesis check is `scripts/verify_core_synthesis.tcl`. With
Vivado 2022.2, the revised core synthesizes to 24,914 LUTs, 26,104 registers
and four DSPs. Its out-of-context 4 ns clock estimate has +0.994 ns setup
slack. These are core synthesis estimates, not routed shell or board results.
The complete RTL replay passed 13,800 flows and 138 windows, including 202
inferences using the test's changed threshold. Four host regression tests pass.

## Build and use a board

```bash
testbed/hardware/u55c/scripts/build_u55c.sh all
testbed/hardware/u55c/scripts/program_u55c.sh --dry-run
# Review timing/route/DRC reports and the proposed image before programming.
testbed/hardware/u55c/scripts/program_u55c.sh
simulation/.venv/bin/python testbed/hardware/u55c/scripts/run_online_adaptation.py \
  --backend jtag --output testbed/results/encoded-mlp-board.json
```

Build requires Vivado, a U55C license and OpenNIC at `OPENNIC_ROOT` (default
`/home/zealot/FTC/third_party/open-nic-shell`). Override `VIVADO_BIN` and
`PYTHON_BIN` if needed. The build uses the new `encoded_mlp_v5` namespace;
timing-closed results go under `build/encoded_mlp_v5/closure/`.
Programming retains stale-image, timing, routing, DRC and bound-driver checks.

The controller requires a freshly reset image, matching ABI/build ID, original
checkpoint and dataset configuration. It chooses the oldest ready window.
Dual buffering permits bounded inference ahead of feedback; this is an
asynchronous hardware schedule, so online updates need not occur at exactly the
same sample as synchronous simulation. Actual per-record model versions are
validated. An update committed after the final prediction is listed under
`unobserved_model_versions`, not counted as a validated updated inference.

Use `--config`, `--checkpoint` and `--assets` consistently for alternative
exports. Verified feedback uses the simulation JSONL format:
`sample_id`, `label`, `verified: true`, `available_window`, with the configured
`feedback.max_labels` budget. Ground truth is used for reporting after inference;
it is never automatically promoted into feedback.

Read-only board status:

```bash
simulation/.venv/bin/python testbed/hardware/u55c/scripts/read_fpga_results.py
```

## Memory and ABI 5

One image is 1,540 bytes:

| Word range | Contents |
|---|---|
| 0–255 | 1,024 evidence-table int8 entries, four per word |
| 256–355 | 400 int8 weights, layer/row order, four per word |
| 356–381 | 26 int32 biases |
| 382–383 | Hidden requantization multipliers |
| 384 | Signed int32 decision threshold |

Byte packing is little endian. The host quantizer's minima/inverse widths add
128 bytes to the simulation inference footprint (1,668 bytes). FPGA update
payloads transfer 1,540 bytes, excluding transport overhead.

Private JTAG AXI-Lite registers:

| Offset | Meaning |
|---|---|
| 0x000 / 0x004 / 0x008 | Feature magic / ABI 0x00050000 / mode 1 |
| 0x00c | Status: ready banks bits 0–1, finished bit 3, loaded bit 4, updating bit 5, model error bit 6, release error bit 7 |
| 0x010 | Write 1: begin image; write 2: commit complete image |
| 0x014 / 0x018 | Completed bank window IDs, one based |
| 0x01c / 0x020 | Completed bank record counts |
| 0x024 / 0x028 | First evaluation sample index per bank |
| 0x034 / 0x038 | Active model version / model word count |
| 0x03c / 0x040 | Captured sample count, low/high |
| 0x044 / 0x048 | Run cycles, low/high; includes startup and host stalls |
| 0x060 | Drift flags for completed banks |
| 0x064 | Sticky errors: stage/incomplete image, invalid bin, arithmetic overflow, invalid release |
| 0x068 | Unique staged word count |
| 0x070 / 0x074 / 0x07c | Window size / evaluation sample count / asset build ID |
| 0x080 / 0x084 | Expected window ID / release bank bitmask |
| 0x4000 + 4i | Write model word i |
| 0x10000 / 0x20000 | Completed bank records, 32 bytes each |

Each record has four words of bin IDs, margin, prediction, model version, and a
reserved word. Read buffers only when their ready bit is set; release with the
matching window ID. AXI writes require all four byte strobes. Unsupported,
unaligned and invalid operations return SLVERR. The public shell aperture exposes
only immutable magic, ABI and mode discovery registers; runtime control uses the
axis-clock JTAG bus.
