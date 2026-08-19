if {$argc != 2} {
    error "usage: verify_u55c.tcl PROJECT REPORT_DIRECTORY"
}
set project_path [file normalize [lindex $argv 0]]
set report_dir [file normalize [lindex $argv 1]]
file mkdir $report_dir
if {![file exists $project_path]} { error "OpenNIC project is missing: $project_path" }

open_project $project_path
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
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {![string match "*Complete*" [get_property STATUS $synth_run]]} {
    error "Synthesis failed: [get_property STATUS $synth_run]"
}

set impl_run [get_runs impl_1]
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {![string match "*Complete*" [get_property STATUS $impl_run]]} {
    error "Implementation failed: [get_property STATUS $impl_run]"
}

open_run impl_1
report_timing_summary -delay_type min_max -max_paths 30 -file [file join $report_dir timing.txt]
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
if {$wns < 0.0 || $whs < 0.0} { error "Timing failed: WNS=$wns WHS=$whs" }
if {![regexp {# of nets with routing errors\.+[[:space:]]*:[[:space:]]*0[[:space:]]*:} $route_report]} {
    error "Routing is incomplete"
}
set drc_errors [get_drc_violations -quiet -filter {SEVERITY == Error}]
if {[llength $drc_errors] != 0} { error "DRC has [llength $drc_errors] errors" }

set bitstream [file join [get_property DIRECTORY $impl_run] open_nic_shell.bit]
if {![file exists $bitstream]} { error "Bitstream is missing: $bitstream" }
puts "DRIFTADAPT_U55C_BITSTREAM=$bitstream"
puts "DRIFTADAPT_U55C_BUILD=PASS"
close_project
