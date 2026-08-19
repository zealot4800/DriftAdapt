# DRIFTADAPT hardware deployment

This directory runs DRIFTADAPT on the Alveo U55C installed in this host. The
U55C implementation is a Vivado 2022.2 OpenNIC design with two 100GbE/RS-FEC
ports and a 16-8-4-2 fixed-point DNN in the PF0 transmit path.

The runtime packet path is:

```text
sender on PF0 -> DNN -> QSFP0 -> external byte-preserving forwarder
                                    -> QSFP1 -> receiver on PF1
```

The forwarder must return the complete Ethernet frame unchanged. In
particular, it must preserve bytes 14 through 48. A direct 100GbE/RS-FEC
loopback between QSFP0 and QSFP1 can be used instead.

## Software environment

Use the Python environment shared with stimulation:

```bash
cd /home/zealot/DriftAdapt/stimulation
source .venv/bin/activate
python -m pip install -r requirements.txt -r ../testbed/requirements.txt

cd ../testbed
set -a
source .env
set +a
python scripts/preflight.py
```

Before programming, preflight is expected to report the U55C XRT personality
(`10ee:505c/505d`) and missing OpenNIC interfaces. That is a safety check, not
a reason to change the PCI IDs in the code.

The model paths and adaptation settings in `.env` match
`stimulation/configs/cic-ids2017.yaml`. Regenerate the packet workload after
changing stimulation preprocessing:

```bash
../stimulation/.venv/bin/python ../stimulation/export_testbed.py \
  --config ../stimulation/configs/cic-ids2017.yaml \
  --output sendrecv/data/intrusion-detection.csv
```

## Build the U55C image

The build consumes the existing OpenNIC checkout at
`/home/zealot/FTC/third_party/open-nic-shell` and does not program hardware:

```bash
cd /home/zealot/DriftAdapt/testbed
bash hardware/u55c/scripts/build_u55c.sh all
```

The command must complete synthesis, implementation, routing, DRC, setup/hold
timing, and bitstream generation. Reports are written to
`hardware/u55c/build/`. The accepted timing-closed image and its final reports
are under `hardware/u55c/build/closure/`. See `hardware/u55c/README.md` for
the BAR2 register map.

Programming is deliberately a separate, confirmed operation because it takes
both U55C PCI functions offline and replaces the currently loaded image. Do
not run the adaptive control plane against an image that does not expose the
`DRFT` feature word at BAR2 offset `0x100000`.

After the build passes, validate the programming inputs with
`hardware/u55c/scripts/bringup_u55c.sh --dry-run`. Running the script without
`--dry-run` requires an explicit interactive confirmation, programs through
JTAG, rescans PCIe, loads the matching OpenNIC driver with RS-FEC, validates
both links, and prints the resolved interface names.

## Configure the programmed card

After programming and PCIe rescan, load the matching OpenNIC module with
RS-FEC enabled. Resolve interface names from the PCI functions instead of
guessing them:

```bash
tx_iface=$(basename /sys/bus/pci/devices/0000:01:00.0/net/*)
rx_iface=$(basename /sys/bus/pci/devices/0000:01:00.1/net/*)
sudo ip link set dev "$rx_iface" up
sudo ip link set dev "$tx_iface" up
sudo ethtool "$tx_iface"
sudo ethtool "$rx_iface"
```

Both links must report `Link detected: yes`. Put the resolved names in
`DRIFTADAPT_TX_IFACE` and `DRIFTADAPT_RX_IFACE`, then rerun preflight. InfluxDB 1.x
must contain the `basic_tcp` measurement used by the adaptive control plane;
use `--skip-influx` only for packet-path bring-up.

The final post-program preflight must run with `sudo --preserve-env` because
the PCI resource is root-readable only. It verifies the `DRFT` feature word,
not just the generic OpenNIC PCI ID.

## Run DRIFTADAPT

Use three terminals after preflight passes:

```bash
# Terminal 1: receive returned predictions
sudo ../stimulation/.venv/bin/python sendrecv/receiver.py \
  --iface "$DRIFTADAPT_RX_IFACE" --logfile receiver-f1.log

# Terminal 2: load weights and run online adaptation
sudo --preserve-env \
  ../stimulation/.venv/bin/python software/intrusion-detection.py

# Terminal 3: inject CIC flows
sudo ../stimulation/.venv/bin/python sendrecv/sender.py \
  --iface "$DRIFTADAPT_TX_IFACE" \
  --datafile sendrecv/data/intrusion-detection.csv
```

Raw packets and BAR access require root. The control plane validates the U55C
feature word before its first register write and commits all 182 Q16.16 model
parameters atomically.
