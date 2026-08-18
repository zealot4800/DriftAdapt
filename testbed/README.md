# Caravan FPGA+P4 Testbed

Here, we provide the code to set up and run the testbed-based evaluation for Caravan. The testbed is based on the [Taurus Anomaly Detection ASPLOS22](https://gitlab.com/dataplane-ai/taurus/applications/anomaly-detection-asplos22) artifact. Hence, when running the intrusion-detection application using our Caravan architecture, we expect the user to have access to a testbed shown in the figure, below.

> **Note for AE reviewers:** This folder is not part of the OSDI 2024 artifact evlaution.

<img src="doc/platform.png" alt="Platform Design" width="600">

* `spatial/`: contains the intrusion-detection spatial code to compile the FPGA bitstream.

* `software/`: contains the control-plane code for monitoring and updating the in-network ML model running on the FPGA.

* `sendrecv/`: contains the code for the packet sender and receiver for traffic generation.

* `open-nic.patch`: needed to patch the Xilinx open-nic-shell---one of the dependencies of the Taurus testbed (referenced above). For running the patch, follow the commands below:
```
cp ./open-nic.patch <path-to-taurus-mapreduce>/deps/open-nic;
cd <path-to-taurus-mapreduce>/deps/open-nic;
git apply open-nic.patch
```

## Components

The testbed has three cooperating processes:

1. `spatial/intrusion-detection.scala` is compiled in the Taurus project and
   implements the 16-8-4-2 fixed-point DNN dataplane on the U250.
2. `software/intrusion-detection.py` reads recent flows from InfluxDB, detects
   an F1 drop, retrains the small DNN, and writes its 182 parameters to the
   accelerator through PCIe BAR1.
3. `sendrecv/sender.py` and `sendrecv/receiver.py` inject labeled packets and
   report the FPGA prediction quality.

## Prepare the U250 host

The repository can be prepared on any machine, but the control plane must run
on the Taurus host with the U250, its driver, `pypci`, and InfluxDB configured.

```bash
cd testbed
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with the interfaces and absolute checkpoint paths, then load it:

```bash
set -a
source .env
set +a
python scripts/preflight.py
```

`preflight.py` is read-only. It checks Python packages, model paths, the Xilinx
PCIe vendor ID, and the InfluxDB TCP endpoint before the control plane accesses
the FPGA.

## Build and configure the dataplane

Copy `spatial/intrusion-detection.scala` into the same application location
used by the Taurus anomaly-detection artifact, apply `open-nic.patch`, and use
that artifact's documented Spatial/OpenNIC build flow to generate and program
the U250 bitstream. The Scala design exposes 182 model parameters and consumes
16 fixed-point packet features. Its output class is written back into the
packet header.

The PCIe addresses in `software/intrusion-detection.py` must match the register
map produced by that build. Do not run the control plane against a differently
generated bitstream.

## Run the testbed

Use separate terminals (and interfaces appropriate for the physical wiring):

```bash
# Terminal 1: collect returned predictions
sudo .venv/bin/python sendrecv/receiver.py \
  --iface "$CARAVAN_RX_IFACE" --logfile receiver-f1.log

# Terminal 2: adaptive control plane on the U250 host
sudo --preserve-env=CARAVAN_INFLUX_HOST,CARAVAN_INFLUX_PORT,CARAVAN_INFLUX_DATABASE,CARAVAN_INITIAL_MODEL,CARAVAN_LABELER_MODEL \
  .venv/bin/python software/intrusion-detection.py

# Terminal 3: send the workload
sudo .venv/bin/python sendrecv/sender.py --iface "$CARAVAN_TX_IFACE" \
  --datafile sendrecv/data/intrusion-detection.csv
```

Run the preflight check again after driver, bitstream, database, interface, or
checkpoint changes. Root privileges are required for raw packets and PCIe;
limit them to the commands that need them.
