# Standalone synthesis sanity check; this is not shell timing closure.
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file join $root build encoded_mlp_v5 core_synthesis]
file mkdir $out
cd [file join $root assets]
create_project -in_memory -part xcu55c-fsvh2892-2L-e
set_property include_dirs [list [file join $root assets]] [current_fileset]
read_verilog -sv [file join $root rtl driftadapt_dnn_axis.sv]
synth_design -top driftadapt_dnn_axis -mode out_of_context -part xcu55c-fsvh2892-2L-e
create_clock -period 4.000 [get_ports clk]
report_utilization -file [file join $out utilization.txt]
report_timing_summary -file [file join $out synthesis_timing.txt]
write_checkpoint -force [file join $out encoded_mlp_core.dcp]
puts "ENCODED_MLP_CORE_SYNTHESIS=PASS"
