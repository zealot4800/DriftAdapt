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
proc signed32 {value} {
    return [expr {$value >= 2147483648 ? $value-4294967296 : $value}]
}
open_hw_manager
connect_hw_server -url $server -allow_non_jtag
open_hw_target
set devices [get_hw_devices -filter {PART =~ *u55c*}]
if {[llength $devices] != 1} {error "Select exactly one U55C JTAG target"}
set device [lindex $devices 0]
current_hw_device $device
refresh_hw_device -update_hw_probes false $device
set axis ""
foreach candidate [get_hw_axis -of_objects $device] {
    if {[string match *driftadapt_jtag_axi_master* [get_property CELL_NAME $candidate]]} {set axis $candidate}
}
if {$axis eq ""} {error "Encoded MLP private JTAG master not found"}
if {[read32 0] != 0x44524654 || [read32 4] != 0x00050000 || [read32 8] != 1} {
    error "Encoded MLP ABI 5 required; rebuild and program the new image"
}
set build [read32 0x7c]
if {$expected_build ne "" && $build != $expected_build} {error "FPGA asset build ID mismatch"}
set result ""
switch -- $operation {
 status {
    set status [read32 0xc]
    set captured [read32 0x3c]
    set version [read32 0x34]
    set errors [read32 0x64]
    set count [read32 0x74]
    set result "\"status\":$status,\"captured\":$captured,\"version\":$version,\"errors\":$errors,\"sample_count\":$count"
 }
 snapshot {
    set bank [lindex $argv 3]
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
    set bank [lindex $argv 3]; set id [lindex $argv 4]
    if {$bank ni {0 1} || $id eq ""} {error "release requires bank and window ID"}
    write32 0x80 $id
    write32 0x84 [expr {1 << $bank}]
    if {[read32 0x64] != 0} {error "Window release rejected"}
    set result "\"released\":$id"
 }
 weights {
    set file [open [lindex $argv 3] r]
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
    set version [read32 0x34]
    for {set retry 0} {$retry < 100 && $version == $old} {incr retry} {
        after 10
        set version [read32 0x34]
    }
    if {$version != $old+1 || [read32 0x64] != 0} {error "Atomic model commit failed"}
    set result "\"version\":$version,\"model_update_bytes\":1540"
 }
 default {error "Unsupported operation: $operation"}
}
puts "DRIFTADAPT_JSON={\"abi\":327680,\"build_id\":$build,$result}"
close_hw_target
disconnect_hw_server
close_hw_manager
