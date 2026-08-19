#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

hardware_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bitstream=${DRIFTADAPT_U55C_BITSTREAM:-$hardware_root/build/closure/driftadapt_u55c_timing_closed.bit}
report_dir=${DRIFTADAPT_U55C_REPORT_DIR:-$hardware_root/build/closure}
driver_module=${DRIFTADAPT_ONIC_MODULE:-/home/zealot/FTC/build/modules/$(uname -r)/onic.ko}
vivado_bin=${VIVADO_BIN:-/home/palhad/Xilinx/Vivado/2022.2/bin/vivado}
program_tcl=${DRIFTADAPT_PROGRAM_TCL:-/home/zealot/FTC/scripts/program_fpga1_jtag.tcl}
bdf0=${DRIFTADAPT_PF0_BDF:-0000:01:00.0}
bdf1=${DRIFTADAPT_PF1_BDF:-0000:01:00.1}
assume_yes=0
dry_run=0
pci_removed=0
bridge_bdf=

usage() {
    cat <<EOF
Usage: $(basename "$0") [--bitstream FILE] [--driver FILE] [--yes] [--dry-run]

Programs the validated DRIFTADAPT OpenNIC image, reloads the matching driver with
RS-FEC, brings up PF0/PF1, and prints the interface settings for testbed/.env.
The operation interrupts both U55C PCI functions and both 100GbE ports.
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo; echo "[$(date '+%F %T')] $*"; }

while (($#)); do
    case "$1" in
        --bitstream) (($# >= 2)) || die "--bitstream requires a file"; bitstream=$2; shift ;;
        --driver) (($# >= 2)) || die "--driver requires a file"; driver_module=$2; shift ;;
        --yes) assume_yes=1 ;;
        --dry-run) dry_run=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
    shift
done

for command_name in sha256sum lspci setpci ip ethtool fuser modinfo insmod rmmod udevadm; do
    command -v "$command_name" >/dev/null 2>&1 || die "required command is missing: $command_name"
done
[[ -s "$bitstream" ]] || die "bitstream is missing or empty: $bitstream"
[[ -r "$driver_module" ]] || die "OpenNIC module is missing: $driver_module"
[[ -x "$vivado_bin" ]] || die "Vivado is missing: $vivado_bin"
[[ -r "$program_tcl" ]] || die "JTAG programming Tcl is missing: $program_tcl"
for report in timing.txt route_status.txt drc.txt; do
    [[ -s "$report_dir/$report" ]] ||
        die "validated implementation report is missing: $report_dir/$report"
done
grep -Fq 'All user specified timing constraints are met.' "$report_dir/timing.txt" ||
    die "timing report does not show timing closure: $report_dir/timing.txt"
grep -Eq '# of nets with routing errors[.]+ :[[:space:]]+0 :' "$report_dir/route_status.txt" ||
    die "route report contains routing errors: $report_dir/route_status.txt"
grep -Fq '| Severity |' "$report_dir/drc.txt" &&
    grep -Eq '\|[^|]+\|[[:space:]]*(Error|Critical Warning)[[:space:]]*\|' "$report_dir/drc.txt" &&
    die "DRC report contains an error or critical warning: $report_dir/drc.txt"
module_kernel=$(modinfo -F vermagic "$driver_module" | awk '{print $1}')
[[ "$module_kernel" == "$(uname -r)" ]] ||
    die "driver was built for $module_kernel, current kernel is $(uname -r)"

bitstream=$(readlink -f "$bitstream")
bit_sha=$(sha256sum "$bitstream" | awk '{print $1}')
echo "Bitstream: $bitstream"
echo "SHA-256:   $bit_sha"
echo "Driver:    $driver_module"
echo "Reports:   $report_dir"
echo "Functions: $bdf0 / $bdf1"

if ((dry_run)); then
    echo "DRIFTADAPT_U55C_DRY_RUN=PASS"
    exit 0
fi

if ((assume_yes == 0)); then
    echo
    echo "This replaces the running U55C image and takes both network ports down."
    read -r -p "Type PROGRAM DRIFTADAPT U55C to continue: " confirmation
    [[ "$confirmation" == "PROGRAM DRIFTADAPT U55C" ]] || die "confirmation did not match"
fi

sudo -v

wait_for_functions() {
    local attempt
    for attempt in {1..30}; do
        [[ -e "/sys/bus/pci/devices/$bdf0" && -e "/sys/bus/pci/devices/$bdf1" ]] && return 0
        sleep 1
    done
    return 1
}

cleanup() {
    local status=$?
    trap - EXIT
    if ((pci_removed)) && [[ -n "$bridge_bdf" ]]; then
        echo "Attempting recovery PCIe rescan..." >&2
        printf '1\n' | sudo tee "/sys/bus/pci/devices/$bridge_bdf/rescan" >/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT

wait_for_functions || die "U55C functions are absent"
path0=$(readlink -f "/sys/bus/pci/devices/$bdf0")
path1=$(readlink -f "/sys/bus/pci/devices/$bdf1")
bridge0=$(basename "$(dirname "$path0")")
bridge1=$(basename "$(dirname "$path1")")
[[ "$bridge0" == "$bridge1" && -e "/sys/bus/pci/devices/$bridge0" ]] ||
    die "the two functions do not share one discoverable upstream bridge"
bridge_bdf=$bridge0

for node in /dev/xclmgmt* /dev/dri/renderD*; do
    [[ -e "$node" ]] || continue
    node_name=$(basename "$node")
    case "$node" in
        /dev/xclmgmt*) node_device=$(readlink -f "/sys/class/xrt_mgmt/$node_name/device" 2>/dev/null || true) ;;
        /dev/dri/*)    node_device=$(readlink -f "/sys/class/drm/$node_name/device" 2>/dev/null || true) ;;
        *)             node_device= ;;
    esac
    node_bdf=$(basename "$node_device")
    [[ "$node_bdf" == "$bdf0" || "$node_bdf" == "$bdf1" ]] || continue
    sudo fuser -s "$node" && {
        sudo fuser -v "$node" || true
        die "U55C/XRT node is in use: $node ($node_bdf)"
    }
done

for bdf in "$bdf0" "$bdf1"; do
    if [[ -d "/sys/bus/pci/devices/$bdf/net" ]]; then
        for interface_path in "/sys/bus/pci/devices/$bdf/net"/*; do
            [[ -e "$interface_path" ]] && sudo ip link set dev "$(basename "$interface_path")" down
        done
    fi
    if [[ -L "/sys/bus/pci/devices/$bdf/driver" ]]; then
        driver_name=$(basename "$(readlink -f "/sys/bus/pci/devices/$bdf/driver")")
        printf '%s\n' "$bdf" | sudo tee "/sys/bus/pci/drivers/$driver_name/unbind" >/dev/null
    fi
done
grep -q '^onic ' /proc/modules && sudo rmmod onic

sudo setpci -s "$bridge_bdf" COMMAND=0000:0100
sudo setpci -s "$bridge_bdf" CAP_EXP+8.w=0000:0004
printf '1\n' | sudo tee "/sys/bus/pci/devices/$bdf1/remove" >/dev/null
printf '1\n' | sudo tee "/sys/bus/pci/devices/$bdf0/remove" >/dev/null
pci_removed=1

note "Programming DRIFTADAPT over JTAG"
env -u LD_LIBRARY_PATH -u PYTHONPATH "$vivado_bin" -mode batch -nojournal -nolog -notrace \
    -source "$program_tcl" -tclargs "$bitstream"

printf '1\n' | sudo tee "/sys/bus/pci/devices/$bridge_bdf/rescan" >/dev/null
wait_for_functions || die "PCI functions did not return after programming"
pci_removed=0

id0="$(<"/sys/bus/pci/devices/$bdf0/vendor"):$(<"/sys/bus/pci/devices/$bdf0/device")"
id1="$(<"/sys/bus/pci/devices/$bdf1/vendor"):$(<"/sys/bus/pci/devices/$bdf1/device")"
[[ "$id0" == "0x10ee:0x903f" && "$id1" == "0x10ee:0x913f" ]] ||
    die "unexpected programmed PCI IDs: $id0 / $id1"

for bdf in "$bdf0" "$bdf1"; do
    sudo setpci -s "$bdf" COMMAND=0x02
    read -r bar_start bar_end _ < <(sed -n '3p' "/sys/bus/pci/devices/$bdf/resource")
    ((16#$bar_start != 0 && 16#$bar_end >= 16#$bar_start && 16#$bar_end - 16#$bar_start + 1 >= 0x400000)) ||
        die "$bdf BAR2 is unassigned; reboot and load the driver afterward"
done

note "Loading OpenNIC with RS-FEC"
sudo insmod "$driver_module" RS_FEC_ENABLED=1
udevadm settle --timeout=10 || die "interface discovery timed out"
tx_iface=$(basename "/sys/bus/pci/devices/$bdf0/net/"*)
rx_iface=$(basename "/sys/bus/pci/devices/$bdf1/net/"*)
[[ -e "/sys/class/net/$tx_iface" && -e "/sys/class/net/$rx_iface" ]] ||
    die "expected one interface on each OpenNIC function"
sudo ip link set dev "$rx_iface" up
sudo ip link set dev "$tx_iface" up

links_ready=0
for attempt in {1..30}; do
    if sudo ethtool "$tx_iface" | grep -q 'Link detected: yes' &&
       sudo ethtool "$rx_iface" | grep -q 'Link detected: yes'; then
        links_ready=1
        break
    fi
    sleep 1
done
((links_ready)) || die "both 100GbE links did not become ready"

echo
echo "DRIFTADAPT_TX_IFACE=$tx_iface"
echo "DRIFTADAPT_RX_IFACE=$rx_iface"
echo "DRIFTADAPT_PCI_RESOURCE=/sys/bus/pci/devices/$bdf0/resource2"
echo "DRIFTADAPT_U55C_BRINGUP=PASS"
