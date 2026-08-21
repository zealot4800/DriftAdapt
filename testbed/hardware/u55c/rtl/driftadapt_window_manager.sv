`timescale 1ns/1ps

// CARAVAN-style window exchange for DRIFTADAPT.
//
// The feature-only source ROM contains no evaluation label. This block records
// those features and the data-plane model's
// prediction in one of two windows.  A control-plane labeling agent reads a
// complete window through AXI-Lite, writes one generated label per sample, and
// submits the bank.  The FPGA then builds the proxy confusion matrix.  The
// control plane uses those counters to calculate macro F1 and decide whether
// retraining is required.
module driftadapt_window_manager #(
    parameter integer DATASET_COUNT = 16000,
    parameter integer WINDOW_SIZE = 100,
    parameter integer PARAM_COUNT = 182,
    parameter integer AXIS_CLOCK_HZ = 250000000,
    parameter integer REFERENCE_WINDOWS = 10
) (
    input  wire          clk,
    input  wire          rst_n,

    input  wire [511:0]  s_axis_tdata,
    input  wire          s_axis_tvalid,
    output wire          s_axis_tready,

    input  wire          generator_active_axis,
    input  wire          generator_finished_axis,
    input  wire [63:0]   sent_packets_axis,
    input  wire [63:0]   classified_packets_axis,
    input  wire [63:0]   elapsed_cycles_axis,
    input  wire [63:0]   input_stall_cycles_axis,
    input  wire [63:0]   output_stall_cycles_axis,
    input  wire [63:0]   classifier_busy_cycles_axis,
    input  wire [63:0]   latency_sum_cycles_axis,
    input  wire [63:0]   latency_min_cycles_axis,
    input  wire [63:0]   latency_max_cycles_axis,
    input  wire [63:0]   total_cycles_axis,
    input  wire          hist_reference_ready_axis,
    input  wire          hist_snapshot_valid_axis,
    input  wire          hist_overflow_error_axis,
    input  wire [31:0]   hist_completed_window_id_axis,
    input  wire [31:0]   hist_completed_window_samples_axis,
    input  wire [8191:0] hist_reference_flat_axis,
    input  wire [8191:0] hist_snapshot0_flat_axis,
    input  wire [8191:0] hist_snapshot1_flat_axis,
    input  wire [255:0]  hist_range_min_flat_axis,
    input  wire [255:0]  hist_range_max_flat_axis,
    output wire          online_complete_axis,
    output wire [63:0]   total_captured_axis,
    output wire [63:0]   total_labeled_axis,
    output wire [63:0]   proxy_true_positive_axis,
    output wire [63:0]   proxy_true_negative_axis,
    output wire [63:0]   proxy_false_positive_axis,
    output wire [63:0]   proxy_false_negative_axis,
    output wire          update_selective_mode_status_axis,
    output wire          update_clone_complete_axis,
    output wire          update_verify_error_axis,
    output wire [31:0]   update_patched_parameters_axis,
    output wire [31:0]   update_bytes_axis,
    output wire [63:0]   update_clone_cycles_axis,
    output wire [63:0]   update_patch_cycles_axis,
    output wire [63:0]   update_commit_cycles_axis,
    output wire [63:0]   update_total_cycles_axis,
    output wire [31:0]   update_old_version_axis,
    output wire [31:0]   update_new_version_axis,

    output reg           weight_stage_valid_axis,
    output reg  [7:0]    weight_stage_index_axis,
    output reg  [31:0]   weight_stage_data_axis,
    output reg           weight_clone_request_axis,
    output reg           selective_update_mode_axis,
    output reg           weight_commit_request_axis,
    input  wire          weight_clone_ack_axis,
    input  wire          weight_clone_busy_axis,
    input  wire          weight_commit_ack_axis,
    input  wire          selective_verify_error_axis,
    input  wire          weights_loaded_axis,
    input  wire          active_weight_bank_axis,
    input  wire [31:0]   model_version_axis,

    input  wire          s_axil_awvalid,
    input  wire [31:0]   s_axil_awaddr,
    output wire          s_axil_awready,
    input  wire          s_axil_wvalid,
    input  wire [31:0]   s_axil_wdata,
    output wire          s_axil_wready,
    output reg           s_axil_bvalid,
    output wire [1:0]    s_axil_bresp,
    input  wire          s_axil_bready,
    input  wire          s_axil_arvalid,
    input  wire [31:0]   s_axil_araddr,
    output wire          s_axil_arready,
    output reg           s_axil_rvalid,
    output reg  [31:0]   s_axil_rdata,
    output wire [1:0]    s_axil_rresp,
    input  wire          s_axil_rready
);
    localparam [31:0] FEATURE_WORD = 32'h44524654; // "DRFT"
    localparam [31:0] VERSION_WORD = 32'h00040001; // online-window ABI v4.1
    localparam [31:0] ONLINE_MODE = 32'd2;
    localparam integer WINDOW_INDEX_WIDTH = $clog2(WINDOW_SIZE + 1);
    localparam integer LABEL_WORDS = (WINDOW_SIZE + 31) / 32;
    localparam integer LABEL_STORAGE_BITS = LABEL_WORDS * 32;

    localparam [31:0] BANK0_LABEL_BASE = 32'h00001000;
    localparam [31:0] BANK0_VALID_BASE = 32'h00001100;
    localparam [31:0] BANK1_LABEL_BASE = 32'h00001200;
    localparam [31:0] BANK1_VALID_BASE = 32'h00001300;
    localparam [31:0] BANK0_PRED_BASE  = 32'h00001400;
    localparam [31:0] BANK1_PRED_BASE  = 32'h00001500;
    localparam [31:0] WEIGHT_BASE      = 32'h00004000;
    localparam [31:0] BANK0_RECORD_BASE = 32'h00010000;
    localparam [31:0] BANK1_RECORD_BASE = 32'h00020000;

    // Each record contains the sixteen signed Q8.8 features. Predictions and
    // identifiers have separate storage so no ground-truth bit enters the
    // online validation path.
    (* ram_style = "distributed" *) reg [255:0] feature_mem0 [0:WINDOW_SIZE-1];
    (* ram_style = "distributed" *) reg [255:0] feature_mem1 [0:WINDOW_SIZE-1];
    reg [LABEL_STORAGE_BITS-1:0] prediction_bits0;
    reg [LABEL_STORAGE_BITS-1:0] prediction_bits1;
    reg [31:0] margin_mem0 [0:WINDOW_SIZE-1];
    reg [31:0] margin_mem1 [0:WINDOW_SIZE-1];
    reg [LABEL_STORAGE_BITS-1:0] agent_label_bits0;
    reg [LABEL_STORAGE_BITS-1:0] agent_label_bits1;
    reg [LABEL_STORAGE_BITS-1:0] agent_valid_bits0;
    reg [LABEL_STORAGE_BITS-1:0] agent_valid_bits1;

    reg bank0_ready;
    reg bank1_ready;
    reg capture_bank;
    reg [WINDOW_INDEX_WIDTH-1:0] capture_count;
    reg [WINDOW_INDEX_WIDTH-1:0] bank0_count;
    reg [WINDOW_INDEX_WIDTH-1:0] bank1_count;
    reg [31:0] bank0_window_id;
    reg [31:0] bank1_window_id;
    reg [31:0] bank0_first_sample;
    reg [31:0] bank1_first_sample;
    reg [31:0] next_window_id;
    reg [31:0] submit_window_id;
    reg [63:0] total_captured;
    reg closed_loop_started;
    reg [63:0] closed_loop_cycles;

    reg scoring;
    reg score_bank;
    reg [WINDOW_INDEX_WIDTH-1:0] score_index;
    reg [WINDOW_INDEX_WIDTH-1:0] score_count;
    reg [31:0] score_tp;
    reg [31:0] score_tn;
    reg [31:0] score_fp;
    reg [31:0] score_fn;
    reg [31:0] score_matches;
    reg label_error;

    reg [31:0] last_window_id;
    reg [31:0] last_window_count;
    reg [31:0] last_true_positive;
    reg [31:0] last_true_negative;
    reg [31:0] last_false_positive;
    reg [31:0] last_false_negative;
    reg [31:0] last_matches;
    reg [31:0] windows_labeled;
    reg [63:0] total_labeled;

    reg [PARAM_COUNT-1:0] weight_stage_mask;
    reg [7:0] weight_stage_count;
    reg [31:0] weight_shadow [0:PARAM_COUNT-1];
    reg update_active;
    reg clone_complete;
    reg update_protocol_error;
    reg last_update_selective;
    reg [31:0] last_patched_parameters;
    reg [31:0] last_update_bytes;
    reg [63:0] clone_cycle_counter;
    reg [63:0] patch_cycle_counter;
    reg [63:0] commit_cycle_counter;
    reg [63:0] total_update_cycle_counter;
    reg [63:0] last_clone_cycles;
    reg [63:0] last_patch_cycles;
    reg [63:0] last_commit_cycles;
    reg [63:0] last_total_update_cycles;
    reg [31:0] update_old_version;
    reg [31:0] update_new_version;

    reg aw_pending;
    reg [31:0] awaddr_pending;
    reg w_pending;
    reg [31:0] wdata_pending;
    reg [16:0] hist_selector;

    integer reset_index;
    integer write_word;
    integer write_parameter;

    function automatic [255:0] packet_features_q8(input [511:0] data);
        integer feature_index;
        begin
            packet_features_q8 = 256'd0;
            for (feature_index = 0; feature_index < 16; feature_index = feature_index + 1)
                packet_features_q8[feature_index*16 +: 16] = {
                    data[(14 + feature_index*2)*8 +: 8],
                    data[(15 + feature_index*2)*8 +: 8]
                };
        end
    endfunction

    function automatic bank_labels_complete(
        input bank,
        input integer sample_count
    );
        integer label_index;
        begin
            bank_labels_complete = sample_count > 0;
            for (label_index = 0; label_index < WINDOW_SIZE; label_index = label_index + 1) begin
                if (label_index < sample_count) begin
                    if (bank ? !agent_valid_bits1[label_index] :
                               !agent_valid_bits0[label_index])
                        bank_labels_complete = 1'b0;
                end
            end
        end
    endfunction

    wire capture_bank_ready = capture_bank ? bank1_ready : bank0_ready;
    assign s_axis_tready = !capture_bank_ready;

    wire score_prediction = score_bank ? prediction_bits1[score_index] :
                                         prediction_bits0[score_index];
    wire score_label = score_bank ? agent_label_bits1[score_index] :
                                    agent_label_bits0[score_index];
    wire current_tp = score_label && score_prediction;
    wire current_tn = !score_label && !score_prediction;
    wire current_fp = !score_label && score_prediction;
    wire current_fn = score_label && !score_prediction;
    wire online_complete = generator_finished_axis &&
                           total_labeled == DATASET_COUNT &&
                           !bank0_ready && !bank1_ready && !scoring;
    assign online_complete_axis = online_complete;
    assign total_captured_axis = total_captured;
    assign total_labeled_axis = total_labeled;
    assign proxy_true_positive_axis = {32'd0, last_true_positive};
    assign proxy_true_negative_axis = {32'd0, last_true_negative};
    assign proxy_false_positive_axis = {32'd0, last_false_positive};
    assign proxy_false_negative_axis = {32'd0, last_false_negative};
    assign update_selective_mode_status_axis = update_active ?
        selective_update_mode_axis : last_update_selective;
    assign update_clone_complete_axis = clone_complete;
    assign update_verify_error_axis = update_protocol_error |
        selective_verify_error_axis;
    assign update_patched_parameters_axis = update_active ?
        {24'd0, weight_stage_count} : last_patched_parameters;
    assign update_bytes_axis = update_active ?
        ({24'd0, weight_stage_count} << 2) : last_update_bytes;
    assign update_clone_cycles_axis = update_active ? clone_cycle_counter : last_clone_cycles;
    assign update_patch_cycles_axis = update_active ? patch_cycle_counter : last_patch_cycles;
    assign update_commit_cycles_axis = update_active ? commit_cycle_counter : last_commit_cycles;
    assign update_total_cycles_axis = update_active ? total_update_cycle_counter : last_total_update_cycles;
    assign update_old_version_axis = update_old_version;
    assign update_new_version_axis = update_new_version;

    assign s_axil_awready = rst_n && !aw_pending && !s_axil_bvalid;
    assign s_axil_wready = rst_n && !w_pending && !s_axil_bvalid;
    assign s_axil_bresp = 2'b00;
    assign s_axil_arready = rst_n && !s_axil_rvalid;
    assign s_axil_rresp = 2'b00;

    function automatic [31:0] read_register(input [31:0] address);
        integer word_number;
        integer parameter_number;
        integer record_index;
        integer record_word;
        begin
            read_register = 32'd0;
            case (address)
                32'h00000000: read_register = FEATURE_WORD;
                32'h00000004: read_register = {
                    22'd0, weight_commit_request_axis, weights_loaded_axis,
                    label_error, scoring, bank1_ready, bank0_ready,
                    online_complete, generator_finished_axis,
                    generator_active_axis, 1'b1
                };
                32'h00000008: read_register = VERSION_WORD;
                32'h0000000c: read_register = DATASET_COUNT;
                32'h00000010: read_register = AXIS_CLOCK_HZ;
                32'h00000014: read_register = ONLINE_MODE;
                32'h00000018: read_register = WINDOW_SIZE;
                32'h0000001c: read_register = 32'd11;
                32'h00000020: read_register = {30'd0, bank1_ready, bank0_ready};
                32'h00000024: read_register = bank0_window_id;
                32'h00000028: read_register = bank0_count;
                32'h0000002c: read_register = bank0_first_sample;
                32'h00000030: read_register = bank1_window_id;
                32'h00000034: read_register = bank1_count;
                32'h00000038: read_register = bank1_first_sample;
                32'h0000003c: read_register = total_captured[31:0];
                32'h00000040: read_register = total_captured[63:32];
                32'h00000050: read_register = last_window_id;
                32'h00000054: read_register = last_window_count;
                32'h00000058: read_register = last_true_positive;
                32'h0000005c: read_register = last_true_negative;
                32'h00000060: read_register = last_false_positive;
                32'h00000064: read_register = last_false_negative;
                32'h00000068: read_register = last_matches;
                32'h0000006c: read_register = windows_labeled;
                32'h00000070: read_register = total_labeled[31:0];
                32'h00000074: read_register = total_labeled[63:32];
                32'h00000078: read_register = model_version_axis;
                32'h0000007c: read_register = weight_stage_count;
                32'h00000080: read_register = classified_packets_axis[31:0];
                32'h00000084: read_register = classified_packets_axis[63:32];
                32'h00000088: read_register = elapsed_cycles_axis[31:0];
                32'h0000008c: read_register = elapsed_cycles_axis[63:32];
                32'h00000090: read_register = classifier_busy_cycles_axis[31:0];
                32'h00000094: read_register = classifier_busy_cycles_axis[63:32];
                32'h00000098: read_register = input_stall_cycles_axis[31:0];
                32'h0000009c: read_register = input_stall_cycles_axis[63:32];
                32'h000000a0: read_register = output_stall_cycles_axis[31:0];
                32'h000000a4: read_register = output_stall_cycles_axis[63:32];
                32'h000000a8: read_register = latency_sum_cycles_axis[31:0];
                32'h000000ac: read_register = latency_sum_cycles_axis[63:32];
                32'h000000b0: read_register = latency_min_cycles_axis[31:0];
                32'h000000b4: read_register = latency_min_cycles_axis[63:32];
                32'h000000b8: read_register = latency_max_cycles_axis[31:0];
                32'h000000bc: read_register = latency_max_cycles_axis[63:32];
                32'h000000c0: read_register = sent_packets_axis[31:0];
                32'h000000c4: read_register = sent_packets_axis[63:32];
                32'h000000c8: read_register = total_cycles_axis[31:0];
                32'h000000cc: read_register = total_cycles_axis[63:32];
                32'h000000d0: read_register = closed_loop_cycles[31:0];
                32'h000000d4: read_register = closed_loop_cycles[63:32];
                32'h000000d8: read_register = {
                    28'd0, hist_overflow_error_axis, 1'b0,
                    hist_reference_ready_axis, hist_snapshot_valid_axis
                };
                32'h000000dc: read_register = hist_completed_window_id_axis;
                32'h000000e0: read_register = hist_completed_window_samples_axis;
                32'h000000e4: read_register = {8'd16, 8'd16, REFERENCE_WINDOWS[15:0]};
                32'h000000e8: read_register = {15'd0, hist_selector};
                32'h000000ec: read_register =
                    hist_reference_flat_axis[((hist_selector[7:0] * 16) + hist_selector[15:8])*32 +: 32];
                32'h000000f0: read_register = hist_selector[16] ?
                    hist_snapshot1_flat_axis[((hist_selector[7:0] * 16) + hist_selector[15:8])*32 +: 32] :
                    hist_snapshot0_flat_axis[((hist_selector[7:0] * 16) + hist_selector[15:8])*32 +: 32];
                32'h000000f4: read_register =
                    {{16{hist_range_min_flat_axis[hist_selector[7:0]*16 + 15]}},
                     hist_range_min_flat_axis[hist_selector[7:0]*16 +: 16]};
                32'h000000f8: read_register =
                    {{16{hist_range_max_flat_axis[hist_selector[7:0]*16 + 15]}},
                     hist_range_max_flat_axis[hist_selector[7:0]*16 +: 16]};
                32'h000000fc: read_register = 32'd4160;
                32'h00000118: read_register = {
                    25'd0, weight_commit_request_axis, update_active,
                    update_protocol_error | selective_verify_error_axis,
                    clone_complete, weight_clone_busy_axis,
                    weight_clone_request_axis, update_selective_mode_status_axis
                };
                32'h0000011c: read_register = update_patched_parameters_axis;
                32'h00000120: read_register = update_bytes_axis;
                32'h00000124: read_register = update_clone_cycles_axis[31:0];
                32'h00000128: read_register = update_clone_cycles_axis[63:32];
                32'h0000012c: read_register = update_patch_cycles_axis[31:0];
                32'h00000130: read_register = update_patch_cycles_axis[63:32];
                32'h00000134: read_register = update_commit_cycles_axis[31:0];
                32'h00000138: read_register = update_commit_cycles_axis[63:32];
                32'h0000013c: read_register = update_total_cycles_axis[31:0];
                32'h00000140: read_register = update_total_cycles_axis[63:32];
                32'h00000144: read_register = update_old_version_axis;
                32'h00000148: read_register = update_new_version_axis;
                32'h0000014c: read_register = model_version_axis;
                32'h0000010c: read_register = {
                    model_version_axis[15:0], 5'd0, active_weight_bank_axis,
                    weight_commit_request_axis, (&weight_stage_mask),
                    weight_stage_count
                };
                32'h00000114: read_register = submit_window_id;
                default: begin
                    if (address >= BANK0_LABEL_BASE &&
                        address < BANK0_LABEL_BASE + LABEL_WORDS*4) begin
                        word_number = (address - BANK0_LABEL_BASE) >> 2;
                        read_register = agent_label_bits0[word_number*32 +: 32];
                    end else if (address >= BANK0_VALID_BASE &&
                                 address < BANK0_VALID_BASE + LABEL_WORDS*4) begin
                        word_number = (address - BANK0_VALID_BASE) >> 2;
                        read_register = agent_valid_bits0[word_number*32 +: 32];
                    end else if (address >= BANK1_LABEL_BASE &&
                                 address < BANK1_LABEL_BASE + LABEL_WORDS*4) begin
                        word_number = (address - BANK1_LABEL_BASE) >> 2;
                        read_register = agent_label_bits1[word_number*32 +: 32];
                    end else if (address >= BANK1_VALID_BASE &&
                                 address < BANK1_VALID_BASE + LABEL_WORDS*4) begin
                        word_number = (address - BANK1_VALID_BASE) >> 2;
                        read_register = agent_valid_bits1[word_number*32 +: 32];
                    end else if (address >= BANK0_PRED_BASE &&
                                 address < BANK0_PRED_BASE + LABEL_WORDS*4) begin
                        word_number = (address - BANK0_PRED_BASE) >> 2;
                        read_register = prediction_bits0[word_number*32 +: 32];
                    end else if (address >= BANK1_PRED_BASE &&
                                 address < BANK1_PRED_BASE + LABEL_WORDS*4) begin
                        word_number = (address - BANK1_PRED_BASE) >> 2;
                        read_register = prediction_bits1[word_number*32 +: 32];
                    end else if (address >= WEIGHT_BASE &&
                                 address < WEIGHT_BASE + PARAM_COUNT*4) begin
                        parameter_number = (address - WEIGHT_BASE) >> 2;
                        read_register = weight_shadow[parameter_number];
                    end else if (address >= BANK0_RECORD_BASE &&
                                 address < BANK0_RECORD_BASE + WINDOW_SIZE*64) begin
                        record_index = (address - BANK0_RECORD_BASE) >> 6;
                        record_word = ((address - BANK0_RECORD_BASE) >> 2) & 15;
                        if (record_word < 8)
                            read_register = feature_mem0[record_index][record_word*32 +: 32];
                        else if (record_word == 8)
                            read_register = bank0_first_sample + record_index;
                        else if (record_word == 9)
                            read_register = {31'd0, prediction_bits0[record_index]};
                        else if (record_word == 10)
                            read_register = margin_mem0[record_index];
                    end else if (address >= BANK1_RECORD_BASE &&
                                 address < BANK1_RECORD_BASE + WINDOW_SIZE*64) begin
                        record_index = (address - BANK1_RECORD_BASE) >> 6;
                        record_word = ((address - BANK1_RECORD_BASE) >> 2) & 15;
                        if (record_word < 8)
                            read_register = feature_mem1[record_index][record_word*32 +: 32];
                        else if (record_word == 8)
                            read_register = bank1_first_sample + record_index;
                        else if (record_word == 9)
                            read_register = {31'd0, prediction_bits1[record_index]};
                        else if (record_word == 10)
                            read_register = margin_mem1[record_index];
                    end
                end
            endcase
        end
    endfunction

    initial begin
        if (WINDOW_SIZE <= 0 || WINDOW_SIZE > 128)
            $fatal(1, "DRIFTADAPT WINDOW_SIZE must be in the range 1..128");
        if (PARAM_COUNT != 182)
            $fatal(1, "DRIFTADAPT hardware model requires 182 parameters");
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            bank0_ready <= 1'b0;
            bank1_ready <= 1'b0;
            capture_bank <= 1'b0;
            capture_count <= 0;
            bank0_count <= 0;
            bank1_count <= 0;
            bank0_window_id <= 0;
            bank1_window_id <= 0;
            bank0_first_sample <= 0;
            bank1_first_sample <= 0;
            next_window_id <= 0;
            submit_window_id <= 0;
            total_captured <= 0;
            closed_loop_started <= 1'b0;
            closed_loop_cycles <= 0;
            prediction_bits0 <= 0;
            prediction_bits1 <= 0;
            agent_label_bits0 <= 0;
            agent_label_bits1 <= 0;
            agent_valid_bits0 <= 0;
            agent_valid_bits1 <= 0;
            scoring <= 1'b0;
            score_bank <= 1'b0;
            score_index <= 0;
            score_count <= 0;
            score_tp <= 0;
            score_tn <= 0;
            score_fp <= 0;
            score_fn <= 0;
            score_matches <= 0;
            label_error <= 1'b0;
            last_window_id <= 0;
            last_window_count <= 0;
            last_true_positive <= 0;
            last_true_negative <= 0;
            last_false_positive <= 0;
            last_false_negative <= 0;
            last_matches <= 0;
            windows_labeled <= 0;
            total_labeled <= 0;
            weight_stage_mask <= 0;
            weight_stage_count <= 0;
            weight_stage_valid_axis <= 1'b0;
            weight_stage_index_axis <= 0;
            weight_stage_data_axis <= 0;
            weight_clone_request_axis <= 1'b0;
            selective_update_mode_axis <= 1'b0;
            weight_commit_request_axis <= 1'b0;
            update_active <= 1'b0;
            clone_complete <= 1'b0;
            update_protocol_error <= 1'b0;
            last_update_selective <= 1'b0;
            last_patched_parameters <= 0;
            last_update_bytes <= 0;
            clone_cycle_counter <= 0;
            patch_cycle_counter <= 0;
            commit_cycle_counter <= 0;
            total_update_cycle_counter <= 0;
            last_clone_cycles <= 0;
            last_patch_cycles <= 0;
            last_commit_cycles <= 0;
            last_total_update_cycles <= 0;
            update_old_version <= 0;
            update_new_version <= 0;
            aw_pending <= 1'b0;
            awaddr_pending <= 0;
            w_pending <= 1'b0;
            wdata_pending <= 0;
            hist_selector <= 0;
            s_axil_bvalid <= 1'b0;
            s_axil_rvalid <= 1'b0;
            s_axil_rdata <= 0;
            for (reset_index = 0; reset_index < PARAM_COUNT; reset_index = reset_index + 1)
                weight_shadow[reset_index] <= 0;
        end else begin
            weight_stage_valid_axis <= 1'b0;

            if (update_active) begin
                total_update_cycle_counter <= total_update_cycle_counter + 1'b1;
                if (selective_update_mode_axis && !clone_complete)
                    clone_cycle_counter <= clone_cycle_counter + 1'b1;
                else if (!weight_commit_request_axis)
                    patch_cycle_counter <= patch_cycle_counter + 1'b1;
                if (weight_commit_request_axis)
                    commit_cycle_counter <= commit_cycle_counter + 1'b1;
            end

            if (weight_clone_request_axis && weight_clone_ack_axis) begin
                weight_clone_request_axis <= 1'b0;
                clone_complete <= 1'b1;
            end

            // sent_packets_axis changes on the first source-to-DNN transfer.
            // It is observed one edge later, so seed with two cycles to retain
            // an inclusive first-input-to-final-label measurement.
            if (!closed_loop_started && sent_packets_axis != 0) begin
                closed_loop_started <= 1'b1;
                closed_loop_cycles <= 64'd2;
            end else if (closed_loop_started && !online_complete) begin
                closed_loop_cycles <= closed_loop_cycles + 1'b1;
            end

            if (s_axis_tvalid && s_axis_tready) begin
                if (!capture_bank) begin
                    feature_mem0[capture_count] <= packet_features_q8(s_axis_tdata);
                    prediction_bits0[capture_count] <= s_axis_tdata[47*8];
                    margin_mem0[capture_count] <= s_axis_tdata[49*8 +: 32];
                    if (capture_count == 0) begin
                        bank0_window_id <= next_window_id;
                        bank0_first_sample <= total_captured[31:0];
                    end
                end else begin
                    feature_mem1[capture_count] <= packet_features_q8(s_axis_tdata);
                    prediction_bits1[capture_count] <= s_axis_tdata[47*8];
                    margin_mem1[capture_count] <= s_axis_tdata[49*8 +: 32];
                    if (capture_count == 0) begin
                        bank1_window_id <= next_window_id;
                        bank1_first_sample <= total_captured[31:0];
                    end
                end
                total_captured <= total_captured + 1'b1;

                if (capture_count == WINDOW_SIZE - 1 ||
                    total_captured == DATASET_COUNT - 1) begin
                    if (!capture_bank) begin
                        bank0_count <= capture_count + 1'b1;
                        bank0_ready <= 1'b1;
                        capture_bank <= 1'b1;
                    end else begin
                        bank1_count <= capture_count + 1'b1;
                        bank1_ready <= 1'b1;
                        capture_bank <= 1'b0;
                    end
                    capture_count <= 0;
                    next_window_id <= next_window_id + 1'b1;
                end else begin
                    capture_count <= capture_count + 1'b1;
                end
            end

            if (scoring) begin
                score_tp <= score_tp + current_tp;
                score_tn <= score_tn + current_tn;
                score_fp <= score_fp + current_fp;
                score_fn <= score_fn + current_fn;
                score_matches <= score_matches + (score_prediction == score_label);
                if (score_index == score_count - 1) begin
                    last_window_id <= score_bank ? bank1_window_id : bank0_window_id;
                    last_window_count <= score_count;
                    last_true_positive <= score_tp + current_tp;
                    last_true_negative <= score_tn + current_tn;
                    last_false_positive <= score_fp + current_fp;
                    last_false_negative <= score_fn + current_fn;
                    last_matches <= score_matches + (score_prediction == score_label);
                    windows_labeled <= windows_labeled + 1'b1;
                    total_labeled <= total_labeled + score_count;
                    scoring <= 1'b0;
                    if (!score_bank) begin
                        bank0_ready <= 1'b0;
                        agent_valid_bits0 <= 0;
                    end else begin
                        bank1_ready <= 1'b0;
                        agent_valid_bits1 <= 0;
                    end
                    if (capture_bank_ready && capture_count == 0)
                        capture_bank <= score_bank;
                end else begin
                    score_index <= score_index + 1'b1;
                end
            end

            if (weight_commit_request_axis && weight_commit_ack_axis) begin
                weight_commit_request_axis <= 1'b0;
                last_update_selective <= selective_update_mode_axis;
                last_patched_parameters <= {24'd0, weight_stage_count};
                last_update_bytes <= {22'd0, weight_stage_count, 2'b00};
                last_clone_cycles <= clone_cycle_counter;
                last_patch_cycles <= patch_cycle_counter;
                last_commit_cycles <= commit_cycle_counter + 1'b1;
                last_total_update_cycles <= total_update_cycle_counter + 1'b1;
                update_new_version <= model_version_axis;
                update_active <= 1'b0;
                clone_complete <= 1'b0;
                selective_update_mode_axis <= 1'b0;
                weight_stage_mask <= 0;
                weight_stage_count <= 0;
            end

            if (s_axil_awvalid && s_axil_awready) begin
                aw_pending <= 1'b1;
                awaddr_pending <= s_axil_awaddr;
            end
            if (s_axil_wvalid && s_axil_wready) begin
                w_pending <= 1'b1;
                wdata_pending <= s_axil_wdata;
            end
            if (aw_pending && w_pending && !s_axil_bvalid) begin
                if (awaddr_pending == 32'h00000100) begin
                    if (wdata_pending[0] && bank0_ready && !scoring) begin
                        if (submit_window_id != bank0_window_id) begin
                            label_error <= 1'b1;
                        end else if (bank_labels_complete(1'b0, bank0_count)) begin
                            scoring <= 1'b1;
                            score_bank <= 1'b0;
                            score_count <= bank0_count;
                            score_index <= 0;
                            score_tp <= 0; score_tn <= 0;
                            score_fp <= 0; score_fn <= 0; score_matches <= 0;
                            label_error <= 1'b0;
                        end else begin
                            label_error <= 1'b1;
                        end
                    end else if (wdata_pending[1] && bank1_ready && !scoring) begin
                        if (submit_window_id != bank1_window_id) begin
                            label_error <= 1'b1;
                        end else if (bank_labels_complete(1'b1, bank1_count)) begin
                            scoring <= 1'b1;
                            score_bank <= 1'b1;
                            score_count <= bank1_count;
                            score_index <= 0;
                            score_tp <= 0; score_tn <= 0;
                            score_fp <= 0; score_fn <= 0; score_matches <= 0;
                            label_error <= 1'b0;
                        end else begin
                            label_error <= 1'b1;
                        end
                    end
                end else if (awaddr_pending == 32'h00000104 && wdata_pending[0]) begin
                    label_error <= 1'b0;
                end else if (awaddr_pending == 32'h00000108 && wdata_pending[0]) begin
                    if ((selective_update_mode_axis && clone_complete &&
                         weight_stage_count != 0) ||
                        (!selective_update_mode_axis && (&weight_stage_mask)))
                        weight_commit_request_axis <= 1'b1;
                    else
                        update_protocol_error <= 1'b1;
                end else if (awaddr_pending == 32'h00000114) begin
                    submit_window_id <= wdata_pending;
                end else if (awaddr_pending == 32'h00000118) begin
                    if (wdata_pending[2]) begin
                        update_protocol_error <= 1'b0;
                    end else if (!update_active && wdata_pending[0]) begin
                        update_active <= 1'b1;
                        selective_update_mode_axis <= 1'b1;
                        weight_clone_request_axis <= 1'b1;
                        clone_complete <= 1'b0;
                        update_protocol_error <= 1'b0;
                        weight_stage_mask <= 0;
                        weight_stage_count <= 0;
                        clone_cycle_counter <= 0;
                        patch_cycle_counter <= 0;
                        commit_cycle_counter <= 0;
                        total_update_cycle_counter <= 0;
                        update_old_version <= model_version_axis;
                        update_new_version <= model_version_axis;
                    end else if (!update_active && wdata_pending[1]) begin
                        update_active <= 1'b1;
                        selective_update_mode_axis <= 1'b0;
                        weight_clone_request_axis <= 1'b0;
                        clone_complete <= 1'b0;
                        update_protocol_error <= 1'b0;
                        weight_stage_mask <= 0;
                        weight_stage_count <= 0;
                        clone_cycle_counter <= 0;
                        patch_cycle_counter <= 0;
                        commit_cycle_counter <= 0;
                        total_update_cycle_counter <= 0;
                        update_old_version <= model_version_axis;
                        update_new_version <= model_version_axis;
                    end else begin
                        update_protocol_error <= 1'b1;
                    end
                end else if (awaddr_pending == 32'h000000e8) begin
                    if (wdata_pending[7:0] < 16 && wdata_pending[15:8] < 16)
                        hist_selector <= wdata_pending[16:0];
                end else if (awaddr_pending >= BANK0_LABEL_BASE &&
                             awaddr_pending < BANK0_LABEL_BASE + LABEL_WORDS*4) begin
                    write_word = (awaddr_pending - BANK0_LABEL_BASE) >> 2;
                    agent_label_bits0[write_word*32 +: 32] <= wdata_pending;
                end else if (awaddr_pending >= BANK0_VALID_BASE &&
                             awaddr_pending < BANK0_VALID_BASE + LABEL_WORDS*4) begin
                    write_word = (awaddr_pending - BANK0_VALID_BASE) >> 2;
                    agent_valid_bits0[write_word*32 +: 32] <= wdata_pending;
                end else if (awaddr_pending >= BANK1_LABEL_BASE &&
                             awaddr_pending < BANK1_LABEL_BASE + LABEL_WORDS*4) begin
                    write_word = (awaddr_pending - BANK1_LABEL_BASE) >> 2;
                    agent_label_bits1[write_word*32 +: 32] <= wdata_pending;
                end else if (awaddr_pending >= BANK1_VALID_BASE &&
                             awaddr_pending < BANK1_VALID_BASE + LABEL_WORDS*4) begin
                    write_word = (awaddr_pending - BANK1_VALID_BASE) >> 2;
                    agent_valid_bits1[write_word*32 +: 32] <= wdata_pending;
                end else if (awaddr_pending >= WEIGHT_BASE &&
                             awaddr_pending < WEIGHT_BASE + PARAM_COUNT*4) begin
                    write_parameter = (awaddr_pending - WEIGHT_BASE) >> 2;
                    if (!selective_update_mode_axis || clone_complete) begin
                        weight_shadow[write_parameter] <= wdata_pending;
                        weight_stage_valid_axis <= 1'b1;
                        weight_stage_index_axis <= write_parameter[7:0];
                        weight_stage_data_axis <= wdata_pending;
                        if (!weight_stage_mask[write_parameter]) begin
                            weight_stage_mask[write_parameter] <= 1'b1;
                            weight_stage_count <= weight_stage_count + 1'b1;
                        end
                    end else begin
                        update_protocol_error <= 1'b1;
                    end
                end
                aw_pending <= 1'b0;
                w_pending <= 1'b0;
                s_axil_bvalid <= 1'b1;
            end else if (s_axil_bvalid && s_axil_bready) begin
                s_axil_bvalid <= 1'b0;
            end

            if (s_axil_arvalid && s_axil_arready) begin
                s_axil_rdata <= read_register(s_axil_araddr);
                s_axil_rvalid <= 1'b1;
            end else if (s_axil_rvalid && s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
            end
        end
    end
endmodule
