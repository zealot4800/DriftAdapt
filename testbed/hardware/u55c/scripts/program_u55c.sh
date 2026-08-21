#!/usr/bin/env bash
set -Eeuo pipefail

hardware_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bitstream=${DRIFTADAPT_U55C_BITSTREAM:-$hardware_root/build/closure/driftadapt_u55c_timing_closed.bit}
report_dir=${DRIFTADAPT_U55C_REPORT_DIR:-$hardware_root/build/closure}
vivado_bin=${VIVADO_BIN:-/home/palhad/Xilinx/Vivado/2022.2/bin/vivado}
program_tcl=${DRIFTADAPT_PROGRAM_TCL:-/home/zealot/FTC/scripts/program_fpga1_jtag.tcl}
assume_yes=0
dry_run=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [--bitstream FILE] [--yes] [--dry-run]

Programs the DRIFTADAPT online-window image over JTAG. Traffic inference and
window capture run in FPGA fabric; a local labeling agent later exchanges
windows through private JTAG AXI. This command does not load a driver,
configure QDMA/CMAC, create Linux interfaces, or use an optical cable.
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

while (($#)); do
    case "$1" in
        --bitstream) (($# >= 2)) || die "--bitstream requires a file"; bitstream=$2; shift ;;
        --yes) assume_yes=1 ;;
        --dry-run) dry_run=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
    shift
done

for command_name in sha256sum grep find; do
    command -v "$command_name" >/dev/null 2>&1 || die "required command is missing: $command_name"
done
[[ -s "$bitstream" ]] || die "bitstream is missing or empty: $bitstream"
[[ -x "$vivado_bin" ]] || die "Vivado is missing: $vivado_bin"
[[ -r "$program_tcl" ]] || die "JTAG programming Tcl is missing: $program_tcl"

# Refuse a timing-closed image left by an older datapath revision. This is
# especially important because build/ is ignored and survives source edits.
while IFS= read -r design_source; do
    [[ "$bitstream" -nt "$design_source" ]] ||
        die "bitstream is stale relative to $design_source; run build_u55c.sh all"
done < <(find "$hardware_root/rtl" "$hardware_root/plugin" "$hardware_root/assets" \
    -type f \( -name '*.sv' -o -name '*.tcl' -o -name '*.mem' -o -name '*.json' \) \
    -print)
for report in timing.txt route_status.txt drc.txt; do
    [[ -s "$report_dir/$report" ]] ||
        die "validated implementation report is missing: $report_dir/$report"
done
grep -Fq 'All user specified timing constraints are met.' "$report_dir/timing.txt" ||
    die "timing report does not show timing closure: $report_dir/timing.txt"
grep -Eq '# of nets with routing errors[.]+ :[[:space:]]+0 :' "$report_dir/route_status.txt" ||
    die "route report contains routing errors: $report_dir/route_status.txt"
if grep -Eq '\|[^|]+\|[[:space:]]*(Error|Critical Warning)[[:space:]]*\|' "$report_dir/drc.txt"; then
    die "DRC report contains an error or critical warning: $report_dir/drc.txt"
fi

# Reprogramming a bound PCI function underneath a host driver is unsafe. A
# normal module unload is sufficient; a reboot is only needed if the module
# refuses to unload cleanly.
if grep -q '^onic ' /proc/modules; then
    die "OpenNIC is loaded; run 'sudo modprobe -r onic' before programming"
fi
for bdf in 0000:01:00.0 0000:01:00.1; do
    if [[ -L "/sys/bus/pci/devices/$bdf/driver" ]]; then
        driver=$(basename "$(readlink -f "/sys/bus/pci/devices/$bdf/driver")")
        die "$bdf is still bound to $driver; unbind/unload it before programming"
    fi
done

bitstream=$(readlink -f "$bitstream")
echo "Bitstream: $bitstream"
echo "SHA-256:   $(sha256sum "$bitstream" | awk '{print $1}')"
echo "Reports:   $report_dir"
echo "Mode:      FPGA inference with local-agent labeling windows"

if ((dry_run)); then
    echo "DRIFTADAPT_U55C_ONLINE_DRY_RUN=PASS"
    exit 0
fi

if ((assume_yes == 0)); then
    echo
    echo "No optical cable or Linux network packet interface is used."
    echo "This replaces the running U55C image; PCI functions must be unbound."
    read -r -p "Type PROGRAM INTERNAL DRIFTADAPT U55C to continue: " confirmation
    [[ "$confirmation" == "PROGRAM INTERNAL DRIFTADAPT U55C" ]] ||
        die "confirmation did not match"
fi

env -u LD_LIBRARY_PATH -u PYTHONPATH \
    "$vivado_bin" -mode batch -nojournal -nolog -notrace \
    -source "$program_tcl" -tclargs "$bitstream"

echo
echo "DRIFTADAPT_U55C_ONLINE_PROGRAM=PASS"
echo "The FPGA now fills two labeling windows and waits for the local agent."
echo "Run hardware/u55c/scripts/run_online_adaptation.py to label, detect drift, and adapt."
