if {$argc != 2} {
    error "usage: close_timing_u55c.tcl ROUTED_DCP OUTPUT_DIRECTORY"
}

set routed_dcp [file normalize [lindex $argv 0]]
set output_dir [file normalize [lindex $argv 1]]
file mkdir $output_dir
if {![file exists $routed_dcp]} { error "Routed checkpoint is missing: $routed_dcp" }

open_checkpoint $routed_dcp
puts "DRIFTADAPT_CLOSURE_INITIAL_WNS=[get_property SLACK [lindex [get_timing_paths -delay_type max -max_paths 1] 0]]"

# The first implementation is fully routed but its remaining setup paths are
# in the stock QDMA aligner. Use Vivado's post-route critical-path replication
# and an aggressive incremental reroute before accepting a hardware image.
phys_opt_design -directive AggressiveExplore
route_design -directive AggressiveExplore
phys_opt_design -directive AggressiveExplore
route_design -directive AggressiveExplore

report_timing_summary -delay_type min_max -max_paths 30 \
    -file [file join $output_dir timing.txt]
report_bus_skew -file [file join $output_dir bus_skew.txt]
report_utilization -file [file join $output_dir utilization.txt]
report_drc -file [file join $output_dir drc.txt]
set route_report [report_route_status -return_string]
set route_file [open [file join $output_dir route_status.txt] w]
puts $route_file $route_report
close $route_file

set setup_path [get_timing_paths -delay_type max -max_paths 1]
set hold_path [get_timing_paths -delay_type min -max_paths 1]
if {[llength $setup_path] == 0 || [llength $hold_path] == 0} {
    error "No setup/hold timing path found after closure"
}
set wns [get_property SLACK [lindex $setup_path 0]]
set whs [get_property SLACK [lindex $hold_path 0]]
puts "DRIFTADAPT_U55C_WNS=$wns"
puts "DRIFTADAPT_U55C_WHS=$whs"
if {$wns < 0.0 || $whs < 0.0} { error "Timing failed: WNS=$wns WHS=$whs" }
if {![regexp {# of nets with routing errors\.+[[:space:]]*:[[:space:]]*0[[:space:]]*:} $route_report]} {
    error "Routing is incomplete"
}
set drc_errors [get_drc_violations -quiet -filter {SEVERITY == Error}]
if {[llength $drc_errors] != 0} { error "DRC has [llength $drc_errors] errors" }

set checkpoint [file join $output_dir driftadapt_u55c_timing_closed.dcp]
set bitstream [file join $output_dir driftadapt_u55c_timing_closed.bit]
write_checkpoint -force $checkpoint
write_bitstream -force $bitstream
puts "DRIFTADAPT_U55C_BITSTREAM=$bitstream"
puts "DRIFTADAPT_U55C_BUILD=PASS"
close_design
