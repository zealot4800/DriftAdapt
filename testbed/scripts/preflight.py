#!/usr/bin/env python3
"""Read-only readiness checks for the DRIFTADAPT FPGA testbed."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path
import socket
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT.parent / "stimulation/checkpoint"
DEFAULT_MODELS = {
    "DRIFTADAPT_INITIAL_MODEL": CHECKPOINT_ROOT / "cic-ids2017-in-network-dnn.pt",
    "DRIFTADAPT_LABELER_MODEL": CHECKPOINT_ROOT / "cic-ids2017-labeler-dnn.pt",
}
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "scapy": "scapy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "influxdb": "influxdb",
    "pypci": "pypci",
}
U250_VENDOR = 0x10EE
U250_APPLICATION_DEVICE_ID = 0x903F
U250_GOLDEN_DEVICE_ID = 0xD004
U55C_OPENNIC_PF0_DEVICE_ID = 0x903F
U55C_XRT_DEVICE_IDS = {0x505C, 0x505D}


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def pass_(self, message: str) -> None:
        print(f"PASS  {message}")

    def warn(self, message: str) -> None:
        print(f"WARN  {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"FAIL  {message}")


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding the environment."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def check_packages(report: Report) -> None:
    for module, package in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module) is None:
            report.fail(f"Python dependency missing: {package}")
        else:
            report.pass_(f"Python dependency available: {package}")


def check_interfaces(report: Report) -> None:
    available = {name for _, name in socket.if_nameindex()}
    for variable in ("DRIFTADAPT_TX_IFACE", "DRIFTADAPT_RX_IFACE"):
        interface = os.getenv(variable)
        if not interface:
            report.fail(f"{variable} is not set")
        elif interface not in available:
            report.fail(f"{variable}={interface!r} does not exist on this host")
        else:
            report.pass_(f"{variable}={interface}")


def check_models(report: Report) -> None:
    for variable, default_path in DEFAULT_MODELS.items():
        raw_path = os.getenv(variable) or str(default_path)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            report.fail(f"{variable} must be an absolute path: {path}")
        elif not path.is_file():
            report.fail(f"{variable} does not point to a file: {path}")
        else:
            report.pass_(f"{variable}={path}")


def check_pci(report: Report) -> None:
    target = os.getenv("DRIFTADAPT_HARDWARE_TARGET", "u55c").lower()
    matches: list[str] = []
    xilinx: list[tuple[str, int]] = []
    for device_path in Path("/sys/bus/pci/devices").glob("*"):
        try:
            vendor = int((device_path / "vendor").read_text().strip(), 16)
            device = int((device_path / "device").read_text().strip(), 16)
        except (OSError, ValueError):
            continue
        if vendor == U250_VENDOR:
            description = f"{device_path.name} [10ee:{device:04x}]"
            xilinx.append((description, device))
            if device == (U55C_OPENNIC_PF0_DEVICE_ID if target == "u55c" else U250_APPLICATION_DEVICE_ID):
                matches.append(description)
    descriptions = [description for description, _ in xilinx]
    if target not in {"u55c", "u250"}:
        report.fail("DRIFTADAPT_HARDWARE_TARGET must be 'u55c' or 'u250'")
        return
    if target == "u55c" and matches:
        resource = Path(os.getenv(
            "DRIFTADAPT_PCI_RESOURCE", "/sys/bus/pci/devices/0000:01:00.0/resource2"
        ))
        try:
            resource_size = resource.stat().st_size
        except OSError as error:
            report.fail(f"U55C OpenNIC BAR2 resource is unavailable: {resource}: {error}")
        else:
            if resource_size <= 0x100004:
                report.fail(f"U55C OpenNIC BAR2 is too small ({resource_size} bytes): {resource}")
            else:
                try:
                    with resource.open("rb", buffering=0) as stream:
                        stream.seek(0x100000)
                        raw_feature = stream.read(4)
                    feature = struct.unpack("<I", raw_feature)[0]
                except PermissionError:
                    report.fail(
                        f"cannot read {resource}; rerun final preflight with sudo "
                        "to validate the DRIFTADAPT feature word"
                    )
                except (OSError, struct.error) as error:
                    report.fail(f"cannot read the U55C DRIFTADAPT feature word: {error}")
                else:
                    if feature != 0x44524654:
                        report.fail(
                            f"U55C BAR2 has feature word {feature:#010x}, expected "
                            "0x44524654 (DRFT); the loaded OpenNIC image is not DRIFTADAPT"
                        )
                    else:
                        report.pass_("programmed DRIFTADAPT U55C PF0 found: " + ", ".join(matches))
        return
    if matches:
        report.pass_("programmed U250 PCIe function found: " + ", ".join(matches))
    elif target == "u55c" and any(device in U55C_XRT_DEVICE_IDS for _, device in xilinx):
        report.fail(
            "U55C XRT platform found, but the DRIFTADAPT OpenNIC image is not programmed: "
            + ", ".join(descriptions)
        )
    elif any(device == U250_GOLDEN_DEVICE_ID for _, device in xilinx):
        report.fail(
            "U250 golden-image function d004 found, but the application bitstream "
            "must be programmed before the control plane can use device 903f"
        )
    elif xilinx:
        report.fail(
            "Xilinx PCIe device(s) found, but IDs do not match the supported "
            f"{target.upper()} image: {', '.join(descriptions)}"
        )
    else:
        report.fail("no Xilinx U250 PCIe function found")


def check_influx(report: Report, timeout: float) -> None:
    host = os.getenv("DRIFTADAPT_INFLUX_HOST", "127.0.0.1")
    raw_port = os.getenv("DRIFTADAPT_INFLUX_PORT", "8086")
    try:
        port = int(raw_port)
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except (OSError, ValueError) as error:
        report.fail(f"cannot connect to InfluxDB at {host}:{raw_port}: {error}")
    else:
        report.pass_(f"InfluxDB TCP endpoint reachable at {host}:{port}")


def check_workload(report: Report) -> None:
    datafile = ROOT / "sendrecv/data/intrusion-detection.csv"
    try:
        with datafile.open(newline="", encoding="utf-8") as stream:
            first_row = next(csv.reader(stream))
        if len(first_row) != 17:
            report.fail(f"workload must have exactly 17 columns; found {len(first_row)}")
            return
        [float(value) for value in first_row]
    except (OSError, StopIteration, ValueError) as error:
        report.fail(f"workload is not readable numeric CSV: {error}")
    else:
        report.pass_(f"workload is readable: {datafile}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--skip-influx", action="store_true", help="skip the TCP reachability check")
    parser.add_argument("--timeout", type=float, default=2.0, help="InfluxDB connection timeout")
    args = parser.parse_args()

    load_dotenv(args.env_file)
    report = Report()
    if sys.version_info < (3, 10):
        report.fail(f"Python 3.10+ required; found {sys.version.split()[0]}")
    else:
        report.pass_(f"Python {sys.version.split()[0]}")
    check_packages(report)
    check_interfaces(report)
    check_models(report)
    check_pci(report)
    check_workload(report)
    if args.skip_influx:
        report.warn("InfluxDB reachability check skipped")
    else:
        check_influx(report, args.timeout)

    print(f"\nPreflight {'failed' if report.failures else 'passed'} with {report.failures} failure(s).")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
