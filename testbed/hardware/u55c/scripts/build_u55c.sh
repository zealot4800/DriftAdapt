#!/usr/bin/env bash
set -euo pipefail

hardware_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
open_nic_root=${OPENNIC_ROOT:-/home/zealot/FTC/third_party/open-nic-shell}
vivado_bin=${VIVADO_BIN:-/home/palhad/Xilinx/Vivado/2022.2/bin/vivado}
mode=${1:-create}
tag=driftadapt_dnn_100g
project="$open_nic_root/build/au55c_${tag}/open_nic_shell/open_nic_shell.xpr"
routed_dcp="$open_nic_root/build/au55c_${tag}/open_nic_shell/open_nic_shell.runs/impl_1/open_nic_shell_routed.dcp"
closure_dir="$hardware_root/build/closure"

if [[ "$mode" != "create" && "$mode" != "implement" && "$mode" != "close" && "$mode" != "all" ]]; then
    echo "usage: $0 [create|implement|close|all]" >&2
    exit 2
fi
[[ -x "$vivado_bin" ]] || { echo "Vivado is missing: $vivado_bin" >&2; exit 1; }
[[ -d "$open_nic_root/script" ]] || { echo "OpenNIC is missing: $open_nic_root" >&2; exit 1; }

if [[ -z "${XILINXD_LICENSE_FILE:-}" && -r /home/palhad/.Xilinx/licenses/Xilinx.lic ]]; then
    export XILINXD_LICENSE_FILE=/home/palhad/.Xilinx/licenses/Xilinx.lic
fi

if [[ "$mode" != "close" ]]; then
    cd "$open_nic_root/script"
    "$vivado_bin" -mode batch -nojournal -nolog -notrace \
        -source build.tcl -tclargs \
        -board au55c -tag "$tag" -overwrite 1 -jobs 8 \
        -synth_ip 0 -impl 0 -post_impl 0 \
        -user_plugin "$hardware_root/plugin" \
        -min_pkt_len 64 -max_pkt_len 9600 \
        -use_phys_func 1 -num_phys_func 2 -num_qdma 1 -num_queue 512 \
        -num_cmac_port 2
fi

if [[ "$mode" == "implement" || "$mode" == "all" ]]; then
    verify_status=0
    "$vivado_bin" -mode batch -nojournal -nolog -notrace \
        -source "$hardware_root/scripts/verify_u55c.tcl" \
        -tclargs "$project" "$hardware_root/build" || verify_status=$?
    if [[ "$mode" == "implement" ]]; then
        exit "$verify_status"
    fi
    [[ -s "$routed_dcp" ]] || {
        echo "implementation did not produce a routed checkpoint" >&2
        exit 1
    }
fi

if [[ "$mode" == "close" || "$mode" == "all" ]]; then
    [[ -s "$routed_dcp" ]] || {
        echo "routed checkpoint is missing: $routed_dcp" >&2
        exit 1
    }
    "$vivado_bin" -mode batch -nojournal -nolog -notrace \
        -source "$hardware_root/scripts/close_timing_u55c.tcl" \
        -tclargs "$routed_dcp" "$closure_dir"
fi
