# Window and model exchange for the DRIFTADAPT private JTAG AXI-Lite master.
#
# Modes:
#   status [wait-seconds]
#   snapshot <auto|0|1>
#   submit <bank> <window-id> <comma-separated-label-words> <comma-separated-valid-words>
#   weights <182-line-hex-memory-file>
#   selective <index,hex-value patch file>

set hw_server_url [lindex $argv 0]
set operation [lindex $argv 1]
if {$hw_server_url eq ""} { set hw_server_url "TCP:localhost:3121" }
if {$operation eq ""} { set operation "status" }

proc driftadapt_read_word {hw_axis address tag} {
    variable driftadapt_txn_sequence
    incr driftadapt_txn_sequence
    set txn_name [format "driftadapt_read_%s_%d" $tag $driftadapt_txn_sequence]
    set txn [create_hw_axi_txn $txn_name $hw_axis \
        -address [format "%08X" $address] -len 1 -type read]
    run_hw_axi $txn
    set value [get_property DATA $txn]
    delete_hw_axi_txn $txn
    return $value
}

proc driftadapt_read_integer {hw_axis address tag} {
    set value_hex [driftadapt_read_word $hw_axis $address $tag]
    scan $value_hex %x value
    return $value
}

proc driftadapt_write_word {hw_axis address value tag} {
    variable driftadapt_txn_sequence
    incr driftadapt_txn_sequence
    set txn_name [format "driftadapt_write_%s_%d" $tag $driftadapt_txn_sequence]
    set txn [create_hw_axi_txn $txn_name $hw_axis \
        -address [format "%08X" $address] \
        -data [format "%08X" $value] -len 1 -type write]
    run_hw_axi $txn
    delete_hw_axi_txn $txn
}

proc driftadapt_read_u64 {hw_axis low_address high_address tag} {
    set low [driftadapt_read_integer $hw_axis $low_address ${tag}_lo]
    set high [driftadapt_read_integer $hw_axis $high_address ${tag}_hi]
    return [expr {wide($low) | (wide($high) << 32)}]
}

proc driftadapt_emit_update_metrics {hw_axis mode} {
    set patched [driftadapt_read_integer $hw_axis 0x11c update_patched]
    set update_bytes [driftadapt_read_integer $hw_axis 0x120 update_bytes]
    set clone_cycles [driftadapt_read_u64 $hw_axis 0x124 0x128 clone_cycles]
    set patch_cycles [driftadapt_read_u64 $hw_axis 0x12c 0x130 patch_cycles]
    set commit_cycles [driftadapt_read_u64 $hw_axis 0x134 0x138 commit_cycles]
    set total_cycles [driftadapt_read_u64 $hw_axis 0x13c 0x140 update_total]
    set old_version [driftadapt_read_integer $hw_axis 0x144 old_version]
    set new_version [driftadapt_read_integer $hw_axis 0x148 new_version]
    puts "DRIFTADAPT_UPDATE_MODE=$mode"
    puts "DRIFTADAPT_UPDATE_TOTAL_PARAMETERS=182"
    puts "DRIFTADAPT_UPDATE_PATCHED_PARAMETERS=$patched"
    puts "DRIFTADAPT_UPDATE_BYTES=$update_bytes"
    puts "DRIFTADAPT_UPDATE_CLONE_CYCLES=$clone_cycles"
    puts "DRIFTADAPT_UPDATE_PATCH_CYCLES=$patch_cycles"
    puts "DRIFTADAPT_UPDATE_COMMIT_CYCLES=$commit_cycles"
    puts "DRIFTADAPT_UPDATE_TOTAL_CYCLES=$total_cycles"
    puts "DRIFTADAPT_UPDATE_OLD_MODEL_VERSION=$old_version"
    puts "DRIFTADAPT_UPDATE_NEW_MODEL_VERSION=$new_version"
}

proc driftadapt_emit_status {hw_axis} {
    set registers {
        FEATURE 0x000 STATUS 0x004 VERSION 0x008 DATASET_COUNT 0x00c
        CLOCK_HZ 0x010 MODE 0x014 WINDOW_SIZE 0x018 RECORD_WORDS 0x01c
        READY_MASK 0x020 BANK0_WINDOW_ID 0x024 BANK0_COUNT 0x028
        BANK0_FIRST_SAMPLE 0x02c BANK1_WINDOW_ID 0x030 BANK1_COUNT 0x034
        BANK1_FIRST_SAMPLE 0x038 CAPTURED_LO 0x03c CAPTURED_HI 0x040
        LAST_WINDOW_ID 0x050 LAST_WINDOW_COUNT 0x054 LAST_TP 0x058
        LAST_TN 0x05c LAST_FP 0x060 LAST_FN 0x064 LAST_MATCHES 0x068
        WINDOWS_LABELED 0x06c LABELED_LO 0x070 LABELED_HI 0x074
        MODEL_VERSION 0x078 WEIGHTS_STAGED 0x07c CLASSIFIED_LO 0x080
        CLASSIFIED_HI 0x084 ELAPSED_LO 0x088 ELAPSED_HI 0x08c
        BUSY_LO 0x090 BUSY_HI 0x094 INPUT_STALL_LO 0x098 INPUT_STALL_HI 0x09c
        OUTPUT_STALL_LO 0x0a0 OUTPUT_STALL_HI 0x0a4
        LATENCY_SUM_LO 0x0a8 LATENCY_SUM_HI 0x0ac
        LATENCY_MIN_LO 0x0b0 LATENCY_MIN_HI 0x0b4
        LATENCY_MAX_LO 0x0b8 LATENCY_MAX_HI 0x0bc
        SENT_LO 0x0c0 SENT_HI 0x0c4 TOTAL_CYCLES_LO 0x0c8 TOTAL_CYCLES_HI 0x0cc
        CLOSED_LOOP_LO 0x0d0 CLOSED_LOOP_HI 0x0d4
        HIST_STATUS 0x0d8 HIST_WINDOW_ID 0x0dc HIST_WINDOW_COUNT 0x0e0
        HIST_CONFIG 0x0e4 HIST_SELECTOR 0x0e8 HIST_STATE_BYTES 0x0fc
        WEIGHT_STATUS 0x10c UPDATE_STATUS 0x118 UPDATE_PATCHED 0x11c
        UPDATE_BYTES 0x120 UPDATE_OLD_VERSION 0x144 UPDATE_NEW_VERSION 0x148
    }
    foreach {name address} $registers {
        set value [driftadapt_read_word $hw_axis $address $name]
        puts "DRIFTADAPT_JTAG_${name}=0x${value}"
    }
}

proc driftadapt_emit_histogram {hw_axis window_id window_count} {
    set status [driftadapt_read_integer $hw_axis 0x0d8 hist_status]
    set latest_window [driftadapt_read_integer $hw_axis 0x0dc hist_window]
    set latest_sample_count [driftadapt_read_integer $hw_axis 0x0e0 hist_count]
    set config [driftadapt_read_integer $hw_axis 0x0e4 hist_config]
    set state_bytes [driftadapt_read_integer $hw_axis 0x0fc hist_state_bytes]
    set feature_count [expr {($config >> 24) & 0xff}]
    set bin_count [expr {($config >> 16) & 0xff}]
    set reference_windows [expr {$config & 0xffff}]
    if {($status & 1) == 0} { error "histogram monitor has no completed snapshot" }
    if {$latest_window < $window_id || $latest_window - $window_id > 1} {
        error "histogram/window snapshot alignment was lost"
    }
    puts "DRIFTADAPT_HIST_STATUS=$status"
    puts "DRIFTADAPT_HIST_COMPLETED_WINDOW_ID=$latest_window"
    puts "DRIFTADAPT_HIST_WINDOW_ID=$window_id"
    puts "DRIFTADAPT_HIST_WINDOW_COUNT=$window_count"
    puts "DRIFTADAPT_HIST_LATEST_WINDOW_COUNT=$latest_sample_count"
    puts "DRIFTADAPT_HIST_FEATURE_COUNT=$feature_count"
    puts "DRIFTADAPT_HIST_BIN_COUNT=$bin_count"
    puts "DRIFTADAPT_HIST_REFERENCE_WINDOWS=$reference_windows"
    puts "DRIFTADAPT_HIST_STATE_BYTES=$state_bytes"
    for {set feature_index 0} {$feature_index < $feature_count} {incr feature_index} {
        set selector [expr {$feature_index | (($window_id & 1) << 16)}]
        driftadapt_write_word $hw_axis 0x0e8 $selector hist_range_selector
        set range_min [driftadapt_read_integer $hw_axis 0x0f4 hist_range_min]
        set range_max [driftadapt_read_integer $hw_axis 0x0f8 hist_range_max]
        puts "DRIFTADAPT_HIST_RANGE=$feature_index,$range_min,$range_max"
        for {set bin_index 0} {$bin_index < $bin_count} {incr bin_index} {
            set selector [expr {$feature_index | ($bin_index << 8) | (($window_id & 1) << 16)}]
            driftadapt_write_word $hw_axis 0x0e8 $selector hist_selector
            set reference [driftadapt_read_integer $hw_axis 0x0ec hist_reference]
            set current [driftadapt_read_integer $hw_axis 0x0f0 hist_current]
            puts "DRIFTADAPT_HIST_RECORD=$feature_index,$bin_index,$reference,$current"
        }
    }
}

set driftadapt_txn_sequence 0
open_hw_manager
connect_hw_server -url $hw_server_url -allow_non_jtag
open_hw_target

set driftadapt_device ""
foreach candidate [get_hw_devices] {
    set part [get_property PART $candidate]
    if {[string match -nocase "*u55c*" $part] ||
        [string match -nocase "*xcu280*" $part]} {
        set driftadapt_device $candidate
        break
    }
}
if {$driftadapt_device eq ""} { error "DRIFTADAPT U55C was not found in the JTAG chain" }

current_hw_device $driftadapt_device
refresh_hw_device -update_hw_probes false $driftadapt_device
set driftadapt_axis ""
set available_axes [get_hw_axis -of_objects $driftadapt_device]
foreach candidate $available_axes {
    if {[string match "*driftadapt_jtag_axi_master*" [get_property CELL_NAME $candidate]]} {
        set driftadapt_axis $candidate
        break
    }
}
if {$driftadapt_axis eq "" && [llength $available_axes] == 1} {
    set driftadapt_axis [lindex $available_axes 0]
}
if {$driftadapt_axis eq ""} { error "private DRIFTADAPT JTAG AXI master was not found" }

set feature [driftadapt_read_integer $driftadapt_axis 0x000 feature]
set version [driftadapt_read_integer $driftadapt_axis 0x008 version]
if {$feature != 0x44524654} { error "DRIFTADAPT feature word mismatch" }
if {$version != 0x00040001} { error "DRIFTADAPT online-window ABI mismatch" }

if {$operation eq "status"} {
    set wait_seconds [lindex $argv 2]
    if {$wait_seconds eq ""} { set wait_seconds 0 }
    if {![string is double -strict $wait_seconds] || $wait_seconds < 0} {
        error "status wait time must be a non-negative number"
    }
    set deadline [expr {[clock milliseconds] + int(1000.0 * $wait_seconds)}]
    while {1} {
        set online_status [driftadapt_read_integer $driftadapt_axis 0x004 online_status]
        if {($online_status & 0x8) != 0 || [clock milliseconds] >= $deadline} { break }
        after 50
    }
    driftadapt_emit_status $driftadapt_axis
} elseif {$operation eq "snapshot"} {
    set bank_argument [lindex $argv 2]
    set ready_mask [driftadapt_read_integer $driftadapt_axis 0x020 ready]
    set online_status [driftadapt_read_integer $driftadapt_axis 0x004 online_status]
    if {$bank_argument eq "" || $bank_argument eq "auto"} {
        if {($ready_mask & 3) == 3} {
            set bank0_id [driftadapt_read_integer $driftadapt_axis 0x024 bank0_id]
            set bank1_id [driftadapt_read_integer $driftadapt_axis 0x030 bank1_id]
            set bank [expr {$bank0_id <= $bank1_id ? 0 : 1}]
        } elseif {$ready_mask & 1} {
            set bank 0
        } elseif {$ready_mask & 2} {
            set bank 1
        } else {
            puts "DRIFTADAPT_WINDOW_READY=0"
            puts "DRIFTADAPT_ONLINE_COMPLETE=[expr {($online_status >> 3) & 1}]"
            close_hw_target
            disconnect_hw_server
            close_hw_manager
            exit
        }
    } else {
        set bank $bank_argument
    }
    if {$bank ni {0 1}} { error "snapshot bank must be auto, 0, or 1" }
    if {(($ready_mask >> $bank) & 1) == 0} {
        error "requested DRIFTADAPT bank is not ready"
    }

    if {$bank == 0} {
        set id_address 0x024
        set count_address 0x028
        set first_address 0x02c
        set record_base 0x10000
    } else {
        set id_address 0x030
        set count_address 0x034
        set first_address 0x038
        set record_base 0x20000
    }
    set window_id [driftadapt_read_integer $driftadapt_axis $id_address window_id]
    set sample_count [driftadapt_read_integer $driftadapt_axis $count_address count]
    set first_sample [driftadapt_read_integer $driftadapt_axis $first_address first]
    puts "DRIFTADAPT_WINDOW_READY=1"
    puts "DRIFTADAPT_ONLINE_COMPLETE=0"
    puts "DRIFTADAPT_WINDOW_BANK=$bank"
    puts "DRIFTADAPT_WINDOW_ID=$window_id"
    puts "DRIFTADAPT_WINDOW_COUNT=$sample_count"
    puts "DRIFTADAPT_WINDOW_FIRST_SAMPLE=$first_sample"
    for {set sample 0} {$sample < $sample_count} {incr sample} {
        set fields {}
        for {set word 0} {$word < 11} {incr word} {
            set address [expr {$record_base + $sample * 64 + $word * 4}]
            lappend fields [driftadapt_read_word $driftadapt_axis $address record]
        }
        puts "DRIFTADAPT_WINDOW_RECORD=$sample,[join $fields ,]"
    }
    driftadapt_emit_histogram $driftadapt_axis $window_id $sample_count
} elseif {$operation eq "submit"} {
    set bank [lindex $argv 2]
    set window_id [lindex $argv 3]
    set label_words [split [lindex $argv 4] ,]
    set valid_words [split [lindex $argv 5] ,]
    set window_size [driftadapt_read_integer $driftadapt_axis 0x018 window_size]
    set expected_words [expr {($window_size + 31) / 32}]
    if {$bank ni {0 1}} { error "submit bank must be 0 or 1" }
    if {![string is integer -strict $window_id] || $window_id < 0} {
        error "submit window ID must be a non-negative integer"
    }
    if {[llength $label_words] != $expected_words ||
        [llength $valid_words] != $expected_words} {
        error "submit requires $expected_words label words and $expected_words valid words"
    }
    if {$bank == 0} {
        set label_base 0x1000
        set valid_base 0x1100
    } else {
        set label_base 0x1200
        set valid_base 0x1300
    }
    for {set word 0} {$word < $expected_words} {incr word} {
        scan [lindex $label_words $word] %x label_value
        scan [lindex $valid_words $word] %x valid_value
        driftadapt_write_word $driftadapt_axis [expr {$label_base + $word*4}] $label_value label
        driftadapt_write_word $driftadapt_axis [expr {$valid_base + $word*4}] $valid_value valid
    }
    set labeled_before [driftadapt_read_integer $driftadapt_axis 0x06c labeled_before]
    driftadapt_write_word $driftadapt_axis 0x114 $window_id submit_window_id
    driftadapt_write_word $driftadapt_axis 0x100 [expr {1 << $bank}] submit
    set deadline [expr {[clock milliseconds] + 5000}]
    while {1} {
        set labeled_now [driftadapt_read_integer $driftadapt_axis 0x06c labeled_now]
        set scored_window [driftadapt_read_integer $driftadapt_axis 0x050 scored_window]
        set submit_status [driftadapt_read_integer $driftadapt_axis 0x004 submit_status]
        if {($submit_status & 0x80) != 0} {
            error "FPGA rejected DRIFTADAPT labels for window $window_id"
        }
        if {$labeled_now > $labeled_before && $scored_window == $window_id} { break }
        if {[clock milliseconds] >= $deadline} { error "timed out scoring DRIFTADAPT labels" }
        after 10
    }
    puts "DRIFTADAPT_SUBMIT_BANK=$bank"
    foreach {name address} {
        WINDOW_ID 0x050 COUNT 0x054 TP 0x058 TN 0x05c FP 0x060 FN 0x064 MATCHES 0x068
    } {
        set value [driftadapt_read_word $driftadapt_axis $address submit_result]
        puts "DRIFTADAPT_SUBMIT_${name}=0x${value}"
    }
} elseif {$operation eq "weights"} {
    set memory_file [lindex $argv 2]
    if {$memory_file eq "" || ![file isfile $memory_file]} {
        error "weights mode requires a readable 182-line memory file"
    }
    set stream [open $memory_file r]
    set contents [split [string trim [read $stream]] "\n"]
    close $stream
    if {[llength $contents] != 182} { error "weight memory must contain exactly 182 words" }
    driftadapt_write_word $driftadapt_axis 0x118 2 full_update_begin
    for {set index 0} {$index < 182} {incr index} {
        scan [string trim [lindex $contents $index]] %x value
        driftadapt_write_word $driftadapt_axis [expr {0x4000 + $index*4}] $value weight
    }
    set staged [driftadapt_read_integer $driftadapt_axis 0x07c staged]
    if {$staged != 182} { error "FPGA accepted $staged of 182 staged parameters" }
    driftadapt_write_word $driftadapt_axis 0x108 1 commit
    set deadline [expr {[clock milliseconds] + 5000}]
    while {1} {
        set status [driftadapt_read_integer $driftadapt_axis 0x10c weight_status]
        if {($status & 0x3ff) == 0} { break }
        if {[clock milliseconds] >= $deadline} { error "timed out committing DRIFTADAPT weights" }
        after 10
    }
    set version [driftadapt_read_integer $driftadapt_axis 0x078 model_version]
    puts "DRIFTADAPT_WEIGHT_COMMIT=PASS"
    puts "DRIFTADAPT_MODEL_VERSION=$version"
    driftadapt_emit_update_metrics $driftadapt_axis full
} elseif {$operation eq "selective"} {
    set patch_file [lindex $argv 2]
    if {$patch_file eq "" || ![file isfile $patch_file]} {
        error "selective mode requires a readable index,value patch file"
    }
    set stream [open $patch_file r]
    set lines [split [string trim [read $stream]] "\n"]
    close $stream
    if {[llength $lines] < 1 || [llength $lines] > 182} {
        error "selective patch count must be in the range 1..182"
    }
    array set seen_indices {}
    set patches {}
    foreach line $lines {
        set fields [split [string trim $line] ,]
        if {[llength $fields] != 2} { error "selective patch lines must be index,hex-value" }
        set parameter_index [string trim [lindex $fields 0]]
        set parameter_hex [string trim [lindex $fields 1]]
        if {![string is integer -strict $parameter_index] ||
            $parameter_index < 0 || $parameter_index >= 182} {
            error "selective parameter index is outside 0..181"
        }
        if {[info exists seen_indices($parameter_index)]} {
            error "duplicate selective parameter index $parameter_index"
        }
        if {[scan $parameter_hex %x parameter_value] != 1} {
            error "invalid selective parameter value: $parameter_hex"
        }
        set seen_indices($parameter_index) 1
        lappend patches [list $parameter_index $parameter_value]
    }

    driftadapt_write_word $driftadapt_axis 0x118 1 selective_clone
    set deadline [expr {[clock milliseconds] + 5000}]
    while {1} {
        set update_status [driftadapt_read_integer $driftadapt_axis 0x118 clone_status]
        if {($update_status & 0x10) != 0} { error "selective clone protocol failed" }
        if {($update_status & 0x8) != 0} { break }
        if {[clock milliseconds] >= $deadline} { error "timed out cloning the active model" }
        after 10
    }

    foreach patch $patches {
        set parameter_index [lindex $patch 0]
        set parameter_value [lindex $patch 1]
        driftadapt_write_word $driftadapt_axis \
            [expr {0x4000 + $parameter_index*4}] $parameter_value selective_patch
    }
    set staged [driftadapt_read_integer $driftadapt_axis 0x07c selective_staged]
    if {$staged != [llength $patches]} {
        error "FPGA accepted $staged of [llength $patches] selective patches"
    }

    driftadapt_write_word $driftadapt_axis 0x108 1 selective_commit
    set deadline [expr {[clock milliseconds] + 5000}]
    while {1} {
        set update_status [driftadapt_read_integer $driftadapt_axis 0x118 verify_status]
        if {($update_status & 0x20) == 0} { break }
        if {[clock milliseconds] >= $deadline} { error "timed out verifying selective update" }
        after 10
    }
    set update_status [driftadapt_read_integer $driftadapt_axis 0x118 final_update_status]
    if {($update_status & 0x10) != 0} {
        error "unchanged shadow parameters differ from the previously active model"
    }
    set old_version [driftadapt_read_integer $driftadapt_axis 0x144 selective_old_version]
    set version [driftadapt_read_integer $driftadapt_axis 0x078 selective_model_version]
    if {$version != $old_version + 1} { error "selective atomic commit did not advance the model version" }
    puts "DRIFTADAPT_SELECTIVE_COMMIT=PASS"
    puts "DRIFTADAPT_MODEL_VERSION=$version"
    driftadapt_emit_update_metrics $driftadapt_axis selective
} else {
    error "unknown DRIFTADAPT online-control operation: $operation"
}

close_hw_target
disconnect_hw_server
close_hw_manager
exit
