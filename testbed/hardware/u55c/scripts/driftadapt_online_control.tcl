# ABI 5 private JTAG control. Only status, snapshot, release and full-model load.
set server [lindex $argv 0]
set operation [lindex $argv 1]
set expected_build [lindex $argv 2]
if {$server eq ""} {set server TCP:localhost:3121}
if {$operation eq ""} {set operation status}
set sequence 0
proc read32 {address} {
    global axis sequence
    set txn [create_hw_axi_txn rd[incr sequence] $axis -address [format %08X $address] -len 1 -type read]
    run_hw_axi $txn
    set hex [get_property DATA $txn]
    delete_hw_axi_txn $txn
    scan $hex %x value
    return $value
}
proc write32 {address value} {
    global axis sequence
    set txn [create_hw_axi_txn wr[incr sequence] $axis -address [format %08X $address] -len 1 -type write -data [format %08X $value]]
    run_hw_axi $txn
    delete_hw_axi_txn $txn
}
# Fault-injection write: never raises; returns 1 if the transaction ran without a
# reported error, 0 otherwise. Callers judge outcomes from registers, not this flag.
proc try_write32 {address value} {
    global axis sequence
    set txn [create_hw_axi_txn fi[incr sequence] $axis -address [format %08X $address] -len 1 -type write -data [format %08X $value]]
    set failed [catch {run_hw_axi $txn} message]
    delete_hw_axi_txn $txn
    return [expr {$failed ? 0 : 1}]
}
proc fi_registers {} {
    return "\"staged_words\":[read32 0x68],\"version\":[read32 0x34],\"errors\":[read32 0x64],\"status\":[read32 0xc],\"captured\":[read32 0x3c]"
}
proc signed32 {value} {
    return [expr {$value >= 2147483648 ? $value-4294967296 : $value}]
}
open_hw_manager
connect_hw_server -url $server -allow_non_jtag
set targets [get_hw_targets -quiet]
if {[llength $targets] == 0} {
    error "No JTAG hardware target is visible at $server. Check board power, the JTAG cable, and which host runs hw_server before starting inference."
}
if {[llength $targets] != 1} {
    error "Expected one JTAG hardware target at $server, found [llength $targets]: $targets. Connect to the hw_server exposing the intended U55C."
}
set target [lindex $targets 0]
current_hw_target $target
open_hw_target $target
set devices [get_hw_devices -filter {PART =~ *u55c*}]
if {[llength $devices] != 1} {error "Select exactly one U55C JTAG target"}
set device [lindex $devices 0]
current_hw_device $device
refresh_hw_device -update_hw_probes false $device
set axis ""
set axes [get_hw_axis -of_objects $device]
foreach candidate $axes {
    if {[string match *driftadapt_jtag_axi_master* [get_property CELL_NAME $candidate]]} {set axis $candidate}
}
if {$axis eq "" && [llength $axes] == 1} {set axis [lindex $axes 0]}
if {$axis eq ""} {error "Encoded MLP private JTAG master not found"}
if {[read32 0] != 0x44524654 || [read32 4] != 0x00050000 || [read32 8] != 1} {
    error "Encoded MLP ABI 5 required; rebuild and program the new image"
}
set build [read32 0x7c]
if {$expected_build ne "" && $build != $expected_build} {error "FPGA asset build ID mismatch"}
proc dispatch {operation args} {
global build
set result ""
switch -- $operation {
 status {
    set status [read32 0xc]
    set captured [read32 0x3c]
    set version [read32 0x34]
    set errors [read32 0x64]
    set count [read32 0x74]
    set cycles [expr {([read32 0x48] << 32) | [read32 0x44]}]
    set result "\"status\":$status,\"captured\":$captured,\"version\":$version,\"errors\":$errors,\"sample_count\":$count,\"cycles\":$cycles"
 }
 snapshot {
    set bank [lindex $args 0]
    set ready [expr {[read32 0xc] & 3}]
    if {$bank eq "" || $bank eq "auto"} {
        if {$ready == 0} {error "No completed window"}
        if {$ready == 3} {set bank [expr {[read32 0x14] < [read32 0x18] ? 0 : 1}]} else {set bank [expr {$ready == 1 ? 0 : 1}]}
    }
    if {$bank ni {0 1} || !($ready & (1 << $bank))} {error "Requested window is not ready"}
    set id [read32 [expr {0x14+4*$bank}]]
    set count [read32 [expr {0x1c+4*$bank}]]
    set first [read32 [expr {0x24+4*$bank}]]
    set drift [expr {([read32 0x60] >> $bank) & 1}]
    set bins {}; set predictions {}; set margins {}; set versions {}
    for {set i 0} {$i < $count} {incr i} {
        set base [expr {($bank+1)*0x10000+$i*32}]
        set row {}
        for {set w 0} {$w < 4} {incr w} {
            set value [read32 [expr {$base+4*$w}]]
            for {set b 0} {$b < 4} {incr b} {lappend row [expr {($value >> (8*$b)) & 255}]}
        }
        lappend bins "\[[join $row ,]\]"
        lappend margins [signed32 [read32 [expr {$base+16}]]]
        lappend predictions [read32 [expr {$base+20}]]
        lappend versions [read32 [expr {$base+24}]]
    }
    if {[read32 [expr {0x14+4*$bank}]] != $id} {error "Window changed during snapshot"}
    set result "\"bank\":$bank,\"window_id\":$id,\"first_sample\":$first,\"drift\":$drift,\"bins\":\[[join $bins ,]\],\"predictions\":\[[join $predictions ,]\],\"margins\":\[[join $margins ,]\],\"versions\":\[[join $versions ,]\]"
 }
 release {
    set bank [lindex $args 0]; set id [lindex $args 1]
    if {$bank ni {0 1} || $id eq ""} {error "release requires bank and window ID"}
    write32 0x80 $id
    write32 0x84 [expr {1 << $bank}]
    if {[read32 0x64] != 0} {error "Window release rejected"}
    set result "\"released\":$id"
 }
 weights {
    set file [open [lindex $args 0] r]
    set words [split [string trim [read $file]] \n]
    close $file
    if {[llength $words] != 385} {error "Model image must have 385 words"}
    foreach word $words {if {![regexp {^[0-9a-fA-F]{8}$} $word]} {error "Malformed model word"}}
    set old [read32 0x34]
    write32 0x10 1
    set i 0
    foreach word $words {scan $word %x value; write32 [expr {0x4000+4*$i}] $value; incr i}
    if {[read32 0x68] != 385 || [read32 0x64] != 0} {error "Model staging failed"}
    write32 0x10 2
    # The swap happens at the next inference boundary. With both window banks
    # full the core is blocked at its output, so the commit legitimately stays
    # pending until the host drains a window; report that instead of failing.
    set version [read32 0x34]
    for {set retry 0} {$retry < 20 && $version == $old} {incr retry} {
        after 10
        set version [read32 0x34]
    }
    if {[read32 0x64] != 0 || ($version != $old && $version != $old+1)} {error "Atomic model commit failed"}
    set committed [expr {$version == $old+1}]
    set result "\"version\":$version,\"committed\":$committed,\"model_update_bytes\":1540"
 }
 fi_write {
    # fi_write ADDRESS VALUE: one raw register write, outcome reported, never fatal.
    set address [lindex $args 0]; set value [lindex $args 1]
    if {$address eq "" || $value eq ""} {error "fi_write requires address and value"}
    set accepted [try_write32 $address $value]
    set result "\"accepted\":$accepted,[fi_registers]"
 }
 fi_stage {
    # fi_stage FILE [N]: stage the first N words of a 385-word image without committing.
    set file [open [lindex $args 0] r]
    set words [split [string trim [read $file]] \n]
    close $file
    set n [lindex $args 1]
    if {$n eq ""} {set n [llength $words]}
    set accepted 0; set i 0
    foreach word [lrange $words 0 [expr {$n-1}]] {
        if {![regexp {^[0-9a-fA-F]{8}$} $word]} {error "Malformed model word"}
        scan $word %x value
        incr accepted [try_write32 [expr {0x4000+4*$i}] $value]
        incr i
    }
    set result "\"written\":$i,\"accepted\":$accepted,[fi_registers]"
 }
 default {error "Unsupported operation: $operation"}
}
puts "DRIFTADAPT_JSON={\"abi\":327680,\"build_id\":$build,$result}"
flush stdout
}
if {$operation eq "serve"} {
    while {[gets stdin request] >= 0} {
        if {$request eq "quit"} {break}
        set fields [split [encoding convertfrom utf-8 [binary format H* $request]] \t]
        if {[catch {dispatch {*}$fields} reason]} {
            binary scan [encoding convertto utf-8 $reason] H* reason_hex
            puts "DRIFTADAPT_ERROR=$reason_hex"
            flush stdout
            break
        }
    }
} else {
    dispatch $operation {*}[lrange $argv 3 end]
}
close_hw_target
disconnect_hw_server
close_hw_manager
