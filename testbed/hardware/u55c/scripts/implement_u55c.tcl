if {$argc != 2} {
    error "usage: implement_u55c.tcl PROJECT REPORT_DIRECTORY"
}
set project_path [file normalize [lindex $argv 0]]
set report_dir [file normalize [lindex $argv 1]]
file mkdir $report_dir
if {![file exists $project_path]} { error "OpenNIC project is missing: $project_path" }

open_project $project_path
# Restore generated-header lookup for projects created by earlier plugin hooks.
set hardware_root [file normalize [file join [file dirname [info script]] ..]]
set source_set [get_filesets sources_1]
set source_includes [get_property include_dirs $source_set]
set asset_dir [file join $hardware_root assets]
if {[lsearch -exact $source_includes $asset_dir] < 0} {
    lappend source_includes $asset_dir
    set_property include_dirs $source_includes $source_set
}
update_compile_order -fileset sources_1
set driftadapt_top [get_files -quiet -compile_order sources -used_in synthesis \
    *driftadapt_u55c_250mhz.sv]
if {[llength $driftadapt_top] != 1} {
    error "DriftAdapt RTL is absent from synthesis compile order; recreate the project with build_u55c.sh create"
}
foreach ip_name {cmac_usplus_0 cmac_usplus_1} {
    set cmac_ip [get_ips -quiet $ip_name]
    if {[llength $cmac_ip] == 0} { error "Missing $ip_name" }
    if {[get_property CONFIG.INCLUDE_RS_FEC $cmac_ip] != 1} {
        error "$ip_name must have RS-FEC enabled"
    }
    if {[get_property CONFIG.INCLUDE_AUTO_NEG_LT_LOGIC $cmac_ip] != 0} {
        error "$ip_name must have auto-negotiation/link-training disabled"
    }
}

set synth_run [get_runs synth_1]
# The plugin sources are referenced in place. Always reset the top run so a
# previous checkpoint cannot hide a changed DRIFTADAPT RTL file.
set_property AUTO_INCREMENTAL_CHECKPOINT 0 $synth_run
set_property INCREMENTAL_CHECKPOINT {} $synth_run
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {![string match "*Complete*" [get_property STATUS $synth_run]]} {
    error "Synthesis failed: [get_property STATUS $synth_run]"
}
open_run synth_1
set core_cell [get_cells -quiet box_250mhz_inst/driftadapt_u55c_inst/dnn]
if {[llength $core_cell] != 1} { error "DriftAdapt core is missing from the synthesized shell" }
report_utilization -cells $core_cell -file [file join $report_dir driftadapt_core_synth.txt]
close_design

set impl_run [get_runs impl_1]
reset_run impl_1
set_property strategy Performance_ExplorePostRoutePhysOpt $impl_run
launch_runs impl_1 -to_step {phys_opt_design (Post-Route)} -jobs 8
wait_on_run impl_1
if {![string match "*Complete*" [get_property STATUS $impl_run]]} {
    error "Implementation failed: [get_property STATUS $impl_run]"
}

open_run impl_1
report_timing_summary -delay_type min_max -max_paths 30 -file [file join $report_dir timing.txt]
report_bus_skew -file [file join $report_dir bus_skew.txt]
report_utilization -file [file join $report_dir utilization.txt]
report_drc -file [file join $report_dir drc.txt]
set route_report [report_route_status -return_string]
set route_file [open [file join $report_dir route_status.txt] w]
puts $route_file $route_report
close $route_file

set setup_path [get_timing_paths -delay_type max -max_paths 1]
set hold_path [get_timing_paths -delay_type min -max_paths 1]
if {[llength $setup_path] == 0 || [llength $hold_path] == 0} {
    error "No implemented setup/hold path found"
}
set wns [get_property SLACK [lindex $setup_path 0]]
set whs [get_property SLACK [lindex $hold_path 0]]
puts "DRIFTADAPT_U55C_WNS=$wns"
puts "DRIFTADAPT_U55C_WHS=$whs"
if {![regexp {# of nets with routing errors\.+[[:space:]]*:[[:space:]]*0[[:space:]]*:} $route_report]} {
    error "Routing is incomplete"
}
set drc_errors [get_drc_violations -quiet -filter {SEVERITY == Error}]
if {[llength $drc_errors] != 0} { error "DRC has [llength $drc_errors] errors" }

set routed_checkpoint [file join [get_property DIRECTORY $impl_run] open_nic_shell_routed.dcp]
write_checkpoint -force $routed_checkpoint
puts "DRIFTADAPT_U55C_ROUTED_CHECKPOINT=$routed_checkpoint"
if {$wns < 0.0 || $whs < 0.0} {
    # Distinct exit status allows 'all' to run post-route optimization, while
    # synthesis, routing and DRC failures still stop the build immediately.
    puts "DRIFTADAPT_U55C_NEEDS_TIMING_CLOSURE=1"
    close_project
    exit 2
}
set bitstream [file join [get_property DIRECTORY $impl_run] open_nic_shell.bit]
write_bitstream -force $bitstream
puts "DRIFTADAPT_U55C_BITSTREAM=$bitstream"
puts "DRIFTADAPT_U55C_BUILD=PASS"
close_project
