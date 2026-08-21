`timescale 1ns/1ps

// Diagnostic mirror on the OpenNIC user AXI-Lite aperture. Only the histogram
// read selector is writable here; the CARAVAN window protocol remains on the
// private JTAG AXI path in driftadapt_window_manager.
module driftadapt_control_regs #(
    parameter integer DATASET_COUNT = 16000,
    parameter integer AXIS_CLOCK_HZ = 250000000,
    parameter integer REFERENCE_WINDOWS = 10
) (
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
    input  wire          s_axil_rready,
    input  wire          axil_aclk,
    input  wire          axil_aresetn,

    input  wire          generator_active_axis,
    input  wire          generator_finished_axis,
    input  wire          result_complete_axis,
    input  wire [63:0]   sent_packets_axis,
    input  wire [63:0]   received_packets_axis,
    input  wire [63:0]   valid_packets_axis,
    input  wire [63:0]   true_positive_axis,
    input  wire [63:0]   true_negative_axis,
    input  wire [63:0]   false_positive_axis,
    input  wire [63:0]   false_negative_axis,
    input  wire [63:0]   malformed_packets_axis,
    input  wire [63:0]   ignored_packets_axis,
    input  wire [63:0]   classified_packets_axis,
    input  wire [63:0]   elapsed_cycles_axis,
    input  wire [63:0]   input_stall_cycles_axis,
    input  wire [63:0]   output_stall_cycles_axis,
    input  wire [63:0]   classifier_busy_cycles_axis,
    input  wire [63:0]   latency_sum_cycles_axis,
    input  wire [63:0]   latency_min_cycles_axis,
    input  wire [63:0]   latency_max_cycles_axis,
    input  wire [63:0]   first_input_cycle_axis,
    input  wire [63:0]   last_output_cycle_axis,
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
    input  wire          update_selective_mode_axis,
    input  wire          update_clone_complete_axis,
    input  wire          update_verify_error_axis,
    input  wire [31:0]   update_patched_parameters_axis,
    input  wire [31:0]   update_bytes_axis,
    input  wire [63:0]   update_clone_cycles_axis,
    input  wire [63:0]   update_patch_cycles_axis,
    input  wire [63:0]   update_commit_cycles_axis,
    input  wire [63:0]   update_total_cycles_axis,
    input  wire [31:0]   update_old_version_axis,
    input  wire [31:0]   update_new_version_axis
);
    localparam [31:0] FEATURE_WORD = 32'h44524654; // "DRFT"
    localparam [31:0] VERSION_WORD = 32'h00040001; // online-window ABI v4.1
    localparam [31:0] BENCHMARK_MODE = 32'h00000002; // hybrid control plane
    localparam [31:0] FEATURE_BYTES = 32'd32; // sixteen signed Q8.8 values
    localparam [31:0] FRAME_BYTES = 32'd60;

    reg aw_pending;
    reg [31:0] awaddr_pending;
    reg w_pending;
    reg [31:0] wdata_pending;
    reg [16:0] hist_selector;

    (* ASYNC_REG = "TRUE" *) reg generator_active_meta, generator_active_axil;
    (* ASYNC_REG = "TRUE" *) reg generator_finished_meta, generator_finished_axil;
    (* ASYNC_REG = "TRUE" *) reg result_complete_meta, result_complete_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] sent_meta, sent_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] received_meta, received_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] valid_meta, valid_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] tp_meta, tp_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] tn_meta, tn_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] fp_meta, fp_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] fn_meta, fn_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] malformed_meta, malformed_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] ignored_meta, ignored_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] classified_meta, classified_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] elapsed_meta, elapsed_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] input_stall_meta, input_stall_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] output_stall_meta, output_stall_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] busy_meta, busy_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] latency_sum_meta, latency_sum_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] latency_min_meta, latency_min_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] latency_max_meta, latency_max_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] first_input_meta, first_input_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] last_output_meta, last_output_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] total_cycles_meta, total_cycles_axil;
    (* ASYNC_REG = "TRUE" *) reg hist_reference_ready_meta, hist_reference_ready_axil;
    (* ASYNC_REG = "TRUE" *) reg hist_snapshot_valid_meta, hist_snapshot_valid_axil;
    (* ASYNC_REG = "TRUE" *) reg hist_overflow_meta, hist_overflow_axil;
    (* ASYNC_REG = "TRUE" *) reg [31:0] hist_window_meta, hist_window_axil;
    (* ASYNC_REG = "TRUE" *) reg [31:0] hist_samples_meta, hist_samples_axil;
    (* ASYNC_REG = "TRUE" *) reg update_selective_meta, update_selective_axil;
    (* ASYNC_REG = "TRUE" *) reg update_clone_meta, update_clone_axil;
    (* ASYNC_REG = "TRUE" *) reg update_error_meta, update_error_axil;
    (* ASYNC_REG = "TRUE" *) reg [31:0] update_patched_meta, update_patched_axil;
    (* ASYNC_REG = "TRUE" *) reg [31:0] update_bytes_meta, update_bytes_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] update_clone_cycles_meta, update_clone_cycles_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] update_patch_cycles_meta, update_patch_cycles_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] update_commit_cycles_meta, update_commit_cycles_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] update_total_cycles_meta, update_total_cycles_axil;
    (* ASYNC_REG = "TRUE" *) reg [31:0] update_old_version_meta, update_old_version_axil;
    (* ASYNC_REG = "TRUE" *) reg [31:0] update_new_version_meta, update_new_version_axil;

    assign s_axil_awready = axil_aresetn && !aw_pending && !s_axil_bvalid;
    assign s_axil_wready = axil_aresetn && !w_pending && !s_axil_bvalid;
    assign s_axil_bresp = 2'b00;
    assign s_axil_arready = axil_aresetn && !s_axil_rvalid;
    assign s_axil_rresp = 2'b00;

    function automatic [31:0] read_register(input [11:0] address);
        begin
            case (address)
                12'h000: read_register = FEATURE_WORD;
                12'h004: read_register = {
                    28'd0, result_complete_axil, generator_finished_axil,
                    generator_active_axil, 1'b1
                };
                12'h008: read_register = VERSION_WORD;
                12'h00c: read_register = DATASET_COUNT;
                12'h010: read_register = AXIS_CLOCK_HZ;
                12'h014: read_register = BENCHMARK_MODE;
                12'h018: read_register = FEATURE_BYTES;
                12'h01c: read_register = FRAME_BYTES;
                12'h020: read_register = sent_axil[31:0];
                12'h024: read_register = sent_axil[63:32];
                12'h028: read_register = received_axil[31:0];
                12'h02c: read_register = received_axil[63:32];
                12'h030: read_register = valid_axil[31:0];
                12'h034: read_register = valid_axil[63:32];
                12'h038: read_register = tp_axil[31:0];
                12'h03c: read_register = tp_axil[63:32];
                12'h040: read_register = tn_axil[31:0];
                12'h044: read_register = tn_axil[63:32];
                12'h048: read_register = fp_axil[31:0];
                12'h04c: read_register = fp_axil[63:32];
                12'h050: read_register = fn_axil[31:0];
                12'h054: read_register = fn_axil[63:32];
                12'h058: read_register = malformed_axil[31:0];
                12'h05c: read_register = malformed_axil[63:32];
                12'h060: read_register = ignored_axil[31:0];
                12'h064: read_register = ignored_axil[63:32];
                12'h068: read_register = classified_axil[31:0];
                12'h06c: read_register = classified_axil[63:32];
                12'h070: read_register = elapsed_axil[31:0];
                12'h074: read_register = elapsed_axil[63:32];
                12'h078: read_register = input_stall_axil[31:0];
                12'h07c: read_register = input_stall_axil[63:32];
                12'h080: read_register = output_stall_axil[31:0];
                12'h084: read_register = output_stall_axil[63:32];
                12'h088: read_register = busy_axil[31:0];
                12'h08c: read_register = busy_axil[63:32];
                12'h090: read_register = latency_sum_axil[31:0];
                12'h094: read_register = latency_sum_axil[63:32];
                12'h098: read_register = latency_min_axil[31:0];
                12'h09c: read_register = latency_min_axil[63:32];
                12'h0a0: read_register = latency_max_axil[31:0];
                12'h0a4: read_register = latency_max_axil[63:32];
                12'h0a8: read_register = first_input_axil[31:0];
                12'h0ac: read_register = first_input_axil[63:32];
                12'h0b0: read_register = last_output_axil[31:0];
                12'h0b4: read_register = last_output_axil[63:32];
                12'h0b8: read_register = total_cycles_axil[31:0];
                12'h0bc: read_register = total_cycles_axil[63:32];
                12'h0d8: read_register = {
                    28'd0, hist_overflow_axil, 1'b0,
                    hist_reference_ready_axil, hist_snapshot_valid_axil
                };
                12'h0dc: read_register = hist_window_axil;
                12'h0e0: read_register = hist_samples_axil;
                12'h0e4: read_register = {8'd16, 8'd16, REFERENCE_WINDOWS[15:0]};
                12'h0e8: read_register = {15'd0, hist_selector};
                12'h0ec: read_register =
                    hist_reference_flat_axis[((hist_selector[7:0] * 16) + hist_selector[15:8])*32 +: 32];
                12'h0f0: read_register = hist_selector[16] ?
                    hist_snapshot1_flat_axis[((hist_selector[7:0] * 16) + hist_selector[15:8])*32 +: 32] :
                    hist_snapshot0_flat_axis[((hist_selector[7:0] * 16) + hist_selector[15:8])*32 +: 32];
                12'h0f4: read_register =
                    {{16{hist_range_min_flat_axis[hist_selector[7:0]*16 + 15]}},
                     hist_range_min_flat_axis[hist_selector[7:0]*16 +: 16]};
                12'h0f8: read_register =
                    {{16{hist_range_max_flat_axis[hist_selector[7:0]*16 + 15]}},
                     hist_range_max_flat_axis[hist_selector[7:0]*16 +: 16]};
                12'h0fc: read_register = 32'd4160;
                12'h118: read_register = {
                    28'd0, update_error_axil, update_clone_axil,
                    1'b0, update_selective_axil
                };
                12'h11c: read_register = update_patched_axil;
                12'h120: read_register = update_bytes_axil;
                12'h124: read_register = update_clone_cycles_axil[31:0];
                12'h128: read_register = update_clone_cycles_axil[63:32];
                12'h12c: read_register = update_patch_cycles_axil[31:0];
                12'h130: read_register = update_patch_cycles_axil[63:32];
                12'h134: read_register = update_commit_cycles_axil[31:0];
                12'h138: read_register = update_commit_cycles_axil[63:32];
                12'h13c: read_register = update_total_cycles_axil[31:0];
                12'h140: read_register = update_total_cycles_axil[63:32];
                12'h144: read_register = update_old_version_axil;
                12'h148: read_register = update_new_version_axil;
                default: read_register = 32'd0;
            endcase
        end
    endfunction

    // Writes are acknowledged; only the bounded histogram selector is retained.
    // Window labels and shadow weights still use the private JTAG block.
    always @(posedge axil_aclk) begin
        if (!axil_aresetn) begin
            aw_pending <= 1'b0;
            awaddr_pending <= 0;
            w_pending <= 1'b0;
            wdata_pending <= 0;
            hist_selector <= 0;
            s_axil_bvalid <= 1'b0;
            s_axil_rvalid <= 1'b0;
            s_axil_rdata <= 32'd0;
        end else begin
            if (s_axil_awvalid && s_axil_awready) begin
                aw_pending <= 1'b1;
                awaddr_pending <= s_axil_awaddr;
            end
            if (s_axil_wvalid && s_axil_wready) begin
                w_pending <= 1'b1;
                wdata_pending <= s_axil_wdata;
            end
            if (aw_pending && w_pending && !s_axil_bvalid) begin
                if (awaddr_pending[11:0] == 12'h0e8 &&
                    wdata_pending[7:0] < 16 && wdata_pending[15:8] < 16)
                    hist_selector <= wdata_pending[16:0];
                aw_pending <= 1'b0;
                w_pending <= 1'b0;
                s_axil_bvalid <= 1'b1;
            end else if (s_axil_bvalid && s_axil_bready) begin
                s_axil_bvalid <= 1'b0;
            end

            if (s_axil_arvalid && s_axil_arready) begin
                s_axil_rdata <= read_register(s_axil_araddr[11:0]);
                s_axil_rvalid <= 1'b1;
            end else if (s_axil_rvalid && s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
            end
        end
    end

    always @(posedge axil_aclk) begin
        if (!axil_aresetn) begin
            generator_active_meta <= 0; generator_active_axil <= 0;
            generator_finished_meta <= 0; generator_finished_axil <= 0;
            result_complete_meta <= 0; result_complete_axil <= 0;
            sent_meta <= 0; sent_axil <= 0;
            received_meta <= 0; received_axil <= 0;
            valid_meta <= 0; valid_axil <= 0;
            tp_meta <= 0; tp_axil <= 0;
            tn_meta <= 0; tn_axil <= 0;
            fp_meta <= 0; fp_axil <= 0;
            fn_meta <= 0; fn_axil <= 0;
            malformed_meta <= 0; malformed_axil <= 0;
            ignored_meta <= 0; ignored_axil <= 0;
            classified_meta <= 0; classified_axil <= 0;
            elapsed_meta <= 0; elapsed_axil <= 0;
            input_stall_meta <= 0; input_stall_axil <= 0;
            output_stall_meta <= 0; output_stall_axil <= 0;
            busy_meta <= 0; busy_axil <= 0;
            latency_sum_meta <= 0; latency_sum_axil <= 0;
            latency_min_meta <= 0; latency_min_axil <= 0;
            latency_max_meta <= 0; latency_max_axil <= 0;
            first_input_meta <= 0; first_input_axil <= 0;
            last_output_meta <= 0; last_output_axil <= 0;
            total_cycles_meta <= 0; total_cycles_axil <= 0;
            hist_reference_ready_meta <= 0; hist_reference_ready_axil <= 0;
            hist_snapshot_valid_meta <= 0; hist_snapshot_valid_axil <= 0;
            hist_overflow_meta <= 0; hist_overflow_axil <= 0;
            hist_window_meta <= 0; hist_window_axil <= 0;
            hist_samples_meta <= 0; hist_samples_axil <= 0;
            update_selective_meta <= 0; update_selective_axil <= 0;
            update_clone_meta <= 0; update_clone_axil <= 0;
            update_error_meta <= 0; update_error_axil <= 0;
            update_patched_meta <= 0; update_patched_axil <= 0;
            update_bytes_meta <= 0; update_bytes_axil <= 0;
            update_clone_cycles_meta <= 0; update_clone_cycles_axil <= 0;
            update_patch_cycles_meta <= 0; update_patch_cycles_axil <= 0;
            update_commit_cycles_meta <= 0; update_commit_cycles_axil <= 0;
            update_total_cycles_meta <= 0; update_total_cycles_axil <= 0;
            update_old_version_meta <= 0; update_old_version_axil <= 0;
            update_new_version_meta <= 0; update_new_version_axil <= 0;
        end else begin
            generator_active_meta <= generator_active_axis;
            generator_active_axil <= generator_active_meta;
            generator_finished_meta <= generator_finished_axis;
            generator_finished_axil <= generator_finished_meta;
            result_complete_meta <= result_complete_axis;
            result_complete_axil <= result_complete_meta;
            sent_meta <= sent_packets_axis; sent_axil <= sent_meta;
            received_meta <= received_packets_axis; received_axil <= received_meta;
            valid_meta <= valid_packets_axis; valid_axil <= valid_meta;
            tp_meta <= true_positive_axis; tp_axil <= tp_meta;
            tn_meta <= true_negative_axis; tn_axil <= tn_meta;
            fp_meta <= false_positive_axis; fp_axil <= fp_meta;
            fn_meta <= false_negative_axis; fn_axil <= fn_meta;
            malformed_meta <= malformed_packets_axis; malformed_axil <= malformed_meta;
            ignored_meta <= ignored_packets_axis; ignored_axil <= ignored_meta;
            classified_meta <= classified_packets_axis; classified_axil <= classified_meta;
            elapsed_meta <= elapsed_cycles_axis; elapsed_axil <= elapsed_meta;
            input_stall_meta <= input_stall_cycles_axis;
            input_stall_axil <= input_stall_meta;
            output_stall_meta <= output_stall_cycles_axis;
            output_stall_axil <= output_stall_meta;
            busy_meta <= classifier_busy_cycles_axis; busy_axil <= busy_meta;
            latency_sum_meta <= latency_sum_cycles_axis;
            latency_sum_axil <= latency_sum_meta;
            latency_min_meta <= latency_min_cycles_axis;
            latency_min_axil <= latency_min_meta;
            latency_max_meta <= latency_max_cycles_axis;
            latency_max_axil <= latency_max_meta;
            first_input_meta <= first_input_cycle_axis;
            first_input_axil <= first_input_meta;
            last_output_meta <= last_output_cycle_axis;
            last_output_axil <= last_output_meta;
            total_cycles_meta <= total_cycles_axis;
            total_cycles_axil <= total_cycles_meta;
            hist_reference_ready_meta <= hist_reference_ready_axis;
            hist_reference_ready_axil <= hist_reference_ready_meta;
            hist_snapshot_valid_meta <= hist_snapshot_valid_axis;
            hist_snapshot_valid_axil <= hist_snapshot_valid_meta;
            hist_overflow_meta <= hist_overflow_error_axis;
            hist_overflow_axil <= hist_overflow_meta;
            hist_window_meta <= hist_completed_window_id_axis;
            hist_window_axil <= hist_window_meta;
            hist_samples_meta <= hist_completed_window_samples_axis;
            hist_samples_axil <= hist_samples_meta;
            update_selective_meta <= update_selective_mode_axis;
            update_selective_axil <= update_selective_meta;
            update_clone_meta <= update_clone_complete_axis;
            update_clone_axil <= update_clone_meta;
            update_error_meta <= update_verify_error_axis;
            update_error_axil <= update_error_meta;
            update_patched_meta <= update_patched_parameters_axis;
            update_patched_axil <= update_patched_meta;
            update_bytes_meta <= update_bytes_axis;
            update_bytes_axil <= update_bytes_meta;
            update_clone_cycles_meta <= update_clone_cycles_axis;
            update_clone_cycles_axil <= update_clone_cycles_meta;
            update_patch_cycles_meta <= update_patch_cycles_axis;
            update_patch_cycles_axil <= update_patch_cycles_meta;
            update_commit_cycles_meta <= update_commit_cycles_axis;
            update_commit_cycles_axil <= update_commit_cycles_meta;
            update_total_cycles_meta <= update_total_cycles_axis;
            update_total_cycles_axil <= update_total_cycles_meta;
            update_old_version_meta <= update_old_version_axis;
            update_old_version_axil <= update_old_version_meta;
            update_new_version_meta <= update_new_version_axis;
            update_new_version_axil <= update_new_version_meta;
        end
    end

endmodule
