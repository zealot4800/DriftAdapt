set plugin_root [file dirname [file normalize [info script]]]
set hardware_root [file normalize [file join $plugin_root ..]]

foreach memory_file {driftadapt_weights.mem driftadapt_samples.mem} {
    set memory_path [file join $hardware_root assets $memory_file]
    if {![file exists $memory_path]} { error "Missing DRIFTADAPT memory image: $memory_path" }
    add_files -fileset sources_1 -norecurse $memory_path
    set_property file_type {Memory Initialization Files} [get_files $memory_path]
}

if {[llength [get_ips -quiet driftadapt_jtag_axi]] == 0} {
    create_ip -name jtag_axi -vendor xilinx.com -library ip -version 1.2 \
        -module_name driftadapt_jtag_axi
}
set_property -dict [list \
    CONFIG.PROTOCOL {2} \
    CONFIG.M_AXI_ADDR_WIDTH {32} \
    CONFIG.M_AXI_DATA_WIDTH {32} \
] [get_ips driftadapt_jtag_axi]
generate_target synthesis [get_ips driftadapt_jtag_axi]
if {[llength [get_runs -quiet driftadapt_jtag_axi_synth_1]] == 0} {
    create_ip_run [get_ips driftadapt_jtag_axi]
}
launch_runs driftadapt_jtag_axi_synth_1
wait_on_run driftadapt_jtag_axi_synth_1

read_verilog -quiet -sv [file join $hardware_root rtl driftadapt_benchmark_metrics.sv]
read_verilog -quiet -sv [file join $hardware_root rtl driftadapt_control_regs.sv]
read_verilog -quiet -sv [file join $hardware_root rtl driftadapt_dnn_axis.sv]
read_verilog -quiet -sv [file join $hardware_root rtl driftadapt_packet_generator.sv]
read_verilog -quiet -sv [file join $hardware_root rtl driftadapt_window_manager.sv]
read_verilog -quiet -sv [file join $plugin_root driftadapt_u55c_250mhz.sv]
