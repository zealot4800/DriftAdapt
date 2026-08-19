set plugin_root [file dirname [file normalize [info script]]]
set hardware_root [file normalize [file join $plugin_root ..]]

read_verilog -quiet -sv [file join $hardware_root rtl driftadapt_control_regs.sv]
read_verilog -quiet -sv [file join $hardware_root rtl driftadapt_dnn_axis.sv]
read_verilog -quiet -sv [file join $plugin_root driftadapt_u55c_250mhz.sv]
