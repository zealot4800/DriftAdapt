# DRIFTADAPT U55C online-learning testbed

The testbed follows CARAVAN's data-plane/control-plane split:

```text
FPGA sample source -> fixed-point DNN -> prediction window A/B
                                           |
                                  private JTAG AXI
                                           |
local labeling agent -> generated labels -> accuracy proxy -> drift trigger
                                                        |
                                                retrain student DNN
                                                        |
                                      shadow weights -> atomic FPGA commit
```

Every completed window contains the sixteen Q8.8 features, a monotonically
increasing sample identifier, the FPGA prediction, and its logit margin. The
local agent produces exactly one binary generated label for every sample. The
FPGA independently reconstructs TP/TN/FP/FN from returned labels and data-plane
predictions; the host verifies those counters and calculates macro F1:

```text
accuracy_proxy[k] = macro_F1(fpga_predictions[k], agent_labels[k])
```

Drift is signaled by a configured temporal proxy drop or by the proxy remaining
below its minimum for consecutive windows. Only then is a class-balanced
student model retrained. All 182 Q16.16 parameters are staged in the inactive
FPGA bank and committed together at an inference boundary.

The committed sample memory contains features only. Dataset labels are used
while generating the offline manifest, but are not written into the synthesized
traffic ROM and cannot influence online validation.

## 1. Regenerate assets when preprocessing or checkpoints change

```bash
cd /home/zealot/DriftAdapt/testbed
../stimulation/.venv/bin/python \
  hardware/u55c/scripts/generate_fpga_assets.py
```

## 2. Build with the existing Vivado 2022.2 installation

```bash
cd /home/zealot/DriftAdapt/testbed
bash hardware/u55c/scripts/build_u55c.sh all
```

The isolated OpenNIC build tag is `driftadapt_online_windows_v4_1`. The validated
image is written to:

```text
hardware/u55c/build/closure/driftadapt_u55c_timing_closed.bit
```

## 3. Program safely

The OpenNIC module must not be bound while the FPGA is replaced:

```bash
sudo modprobe -r onic

cd /home/zealot/DriftAdapt/testbed
bash hardware/u55c/scripts/program_u55c.sh --dry-run
bash hardware/u55c/scripts/program_u55c.sh
```

The FPGA fills at most two windows and then applies backpressure until the
local agent returns labels. No OpenNIC driver, Linux network interface,
InfluxDB service, packet sender/receiver, optical cable, or reboot is required.

## 4. Run the local labeling agent

Do not run the result reader concurrently because both commands use the same
private JTAG AXI master.

```bash
cd /home/zealot/DriftAdapt/testbed
../stimulation/.venv/bin/python -u \
  hardware/u55c/scripts/run_online_adaptation.py \
  --config ../stimulation/configs/cic-ids2017.yaml \
  --device cpu
```

The hardware controller defaults to the local Ollama agent (`--provider local`)
and uses the raw feature values associated with each FPGA sample ID. It reads
only configured feature columns and never reads the CSV ground-truth column.
Before that run, start Ollama and ensure the configured local model is present;
the defaults are `OLLAMA_HOST=http://127.0.0.1:11434` and
`OLLAMA_MODEL=llama2:7b`. The controller loads `stimulation/.env`, and either
setting can also be supplied directly as an environment variable.
For a deterministic labeler smoke run, select the large checkpoint DNN with
`--provider dnn`. In both cases the agent labels every sample.

For a one-window hardware smoke run without retraining:

```bash
../stimulation/.venv/bin/python -u \
  hardware/u55c/scripts/run_online_adaptation.py \
  --provider dnn --max-windows 1 --no-retrain
```

Per-window proxy values, drift decisions, labeling/retraining time, confusion
counters, and model versions are written to `testbed/results/`.

## 5. Read final accuracy-proxy and performance state

After stopping or completing the agent:

```bash
python3 hardware/u55c/scripts/read_fpga_results.py
```

Machine-readable output:

```bash
python3 hardware/u55c/scripts/read_fpga_results.py --json
```

`closed_loop_throughput_samples_per_second` spans the first FPGA input through
the final agent label and includes window backpressure, labeling, and
retraining. `classifier_throughput_samples_per_second` and `mean_latency_ns`
isolate the data-plane DNN. Thus the report separates inference performance
from end-to-end online-adaptation throughput.
