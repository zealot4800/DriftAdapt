`timescale 1ns/1ps

// Label-free, bounded histogram monitor for the sixteen Q8.8 DNN inputs.
// The block observes accepted source packets only and never consumes DNN output,
// labels, or proxy-accuracy state.  JSD is intentionally left to the host.
module driftadapt_hist_monitor #(
    parameter integer FEATURE_COUNT = 16,
    parameter integer BIN_COUNT = 16,
    parameter integer WINDOW_SIZE = 100,
    parameter integer SAMPLE_COUNT = 16000,
    parameter integer REFERENCE_WINDOWS = 10
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire [511:0]                      sample_tdata,
    input  wire                              sample_accept,
    output reg                               reference_ready,
    output reg                               snapshot_valid,
    output reg                               overflow_error,
    output reg  [31:0]                       completed_window_id,
    output reg  [31:0]                       completed_window_samples,
    input  wire [16:0]                       host_read_selector,
    output reg  [31:0]                       host_reference_count,
    output reg  [31:0]                       host_snapshot_count,
    output reg  [31:0]                       host_range_min,
    output reg  [31:0]                       host_range_max,
    input  wire [16:0]                       shell_read_selector,
    output reg  [31:0]                       shell_reference_count,
    output reg  [31:0]                       shell_snapshot_count,
    output reg  [31:0]                       shell_range_min,
    output reg  [31:0]                       shell_range_max
);
    localparam integer HIST_COUNT = FEATURE_COUNT * BIN_COUNT;
    localparam integer WINDOW_COUNT_WIDTH = $clog2(WINDOW_SIZE + 1);

    reg [31:0] reference_hist [0:HIST_COUNT-1];
    reg [31:0] current_hist [0:HIST_COUNT-1];
    reg [31:0] snapshot_hist0 [0:HIST_COUNT-1];
    reg [31:0] snapshot_hist1 [0:HIST_COUNT-1];
    reg signed [15:0] range_min [0:FEATURE_COUNT-1];
    reg signed [15:0] range_max [0:FEATURE_COUNT-1];
    reg [FEATURE_COUNT-1:0] range_valid;
    reg [WINDOW_COUNT_WIDTH-1:0] window_sample_count;
    reg [31:0] next_window_id;
    reg [31:0] total_sample_count;
    reg [255:0] pending_features;
    reg processing_sample;
    reg [3:0] feature_cursor;
    reg [3:0] bin_candidate;
    reg [3:0] selected_bin;
    reg signed [31:0] bin_delta_scaled;
    reg signed [31:0] bin_span;
    reg signed [31:0] bin_threshold;
    reg signed [15:0] prepared_minimum;
    reg signed [15:0] prepared_maximum;
    reg [2:0] feature_state;
    reg [HIST_COUNT-1:0] histogram_write_enable;
    reg [31:0] selected_reference_count;
    reg [31:0] selected_current_count;
    reg pending_window_last;
    reg [31:0] pending_window_samples;
    reg snapshot_copy_active;
    reg snapshot_copy_bank;
    reg [7:0] snapshot_copy_index;
    reg [HIST_COUNT-1:0] snapshot_copy_enable;
    (* ASYNC_REG = "TRUE" *) reg [16:0] shell_selector_meta;
    (* ASYNC_REG = "TRUE" *) reg [16:0] shell_selector_axis;

    localparam [2:0] RANGE_PREPARE = 3'd0;
    localparam [2:0] BIN_PREPARE = 3'd1;
    localparam [2:0] BIN_SEARCH = 3'd2;
    localparam [2:0] WRITE_SELECT = 3'd3;
    localparam [2:0] HISTOGRAM_UPDATE = 3'd4;

    integer reset_index;

    function automatic signed [15:0] packet_feature_q8(
        input [511:0] data,
        input integer index
    );
        begin
            packet_feature_q8 = {
                data[(14 + index*2)*8 +: 8],
                data[(15 + index*2)*8 +: 8]
            };
        end
    endfunction

    function automatic [31:0] saturating_increment(input [31:0] count);
        begin
            saturating_increment = (&count) ? count : count + 1'b1;
        end
    endfunction

    wire signed [15:0] current_feature =
        pending_features[feature_cursor*16 +: 16];
    wire signed [15:0] candidate_minimum =
        !range_valid[feature_cursor] ? current_feature :
        ((current_feature < range_min[feature_cursor]) ? current_feature : range_min[feature_cursor]);
    wire signed [15:0] candidate_maximum =
        !range_valid[feature_cursor] ? current_feature :
        ((current_feature > range_max[feature_cursor]) ? current_feature : range_max[feature_cursor]);
    wire signed [15:0] bin_minimum =
        reference_ready ? range_min[feature_cursor] : candidate_minimum;
    wire signed [15:0] bin_maximum =
        reference_ready ? range_max[feature_cursor] : candidate_maximum;
    wire [7:0] selected_histogram_index = feature_cursor * BIN_COUNT + selected_bin;

    wire [7:0] host_read_index = {host_read_selector[3:0], host_read_selector[11:8]};
    wire [3:0] host_read_feature = host_read_selector[3:0];
    wire [7:0] shell_read_index = {shell_selector_axis[3:0], shell_selector_axis[11:8]};
    wire [3:0] shell_read_feature = shell_selector_axis[3:0];

    initial begin
        if (FEATURE_COUNT != 16 || BIN_COUNT != 16)
            $fatal(1, "DRIFTADAPT histogram monitor requires 16 features and 16 bins");
        if (WINDOW_SIZE <= 0 || SAMPLE_COUNT <= 0 || REFERENCE_WINDOWS <= 0)
            $fatal(1, "DRIFTADAPT histogram configuration must be positive");
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            reference_ready <= 1'b0;
            snapshot_valid <= 1'b0;
            overflow_error <= 1'b0;
            completed_window_id <= 0;
            completed_window_samples <= 0;
            window_sample_count <= 0;
            next_window_id <= 0;
            total_sample_count <= 0;
            range_valid <= 0;
            pending_features <= 0;
            processing_sample <= 1'b0;
            feature_cursor <= 0;
            bin_candidate <= 0;
            selected_bin <= 0;
            bin_delta_scaled <= 0;
            bin_span <= 0;
            bin_threshold <= 0;
            prepared_minimum <= 0;
            prepared_maximum <= 0;
            feature_state <= RANGE_PREPARE;
            histogram_write_enable <= 0;
            selected_reference_count <= 0;
            selected_current_count <= 0;
            pending_window_last <= 1'b0;
            pending_window_samples <= 0;
            snapshot_copy_active <= 1'b0;
            snapshot_copy_bank <= 1'b0;
            snapshot_copy_index <= 0;
            snapshot_copy_enable <= 0;
            shell_selector_meta <= 0;
            shell_selector_axis <= 0;
            host_reference_count <= 0;
            host_snapshot_count <= 0;
            host_range_min <= 0;
            host_range_max <= 0;
            shell_reference_count <= 0;
            shell_snapshot_count <= 0;
            shell_range_min <= 0;
            shell_range_max <= 0;
            for (reset_index = 0; reset_index < HIST_COUNT; reset_index = reset_index + 1) begin
                reference_hist[reset_index] <= 0;
                current_hist[reset_index] <= 0;
                snapshot_hist0[reset_index] <= 0;
                snapshot_hist1[reset_index] <= 0;
            end
            for (reset_index = 0; reset_index < FEATURE_COUNT; reset_index = reset_index + 1) begin
                range_min[reset_index] <= 0;
                range_max[reset_index] <= 0;
            end
        end else begin
            shell_selector_meta <= shell_read_selector;
            shell_selector_axis <= shell_selector_meta;

            // Keep the large counter arrays local.  Only the selected words
            // cross module boundaries, preserving the control-register ABI
            // without routing three flattened 8,192-bit buses.
            host_reference_count <= reference_hist[host_read_index];
            host_snapshot_count <= host_read_selector[16] ?
                snapshot_hist1[host_read_index] : snapshot_hist0[host_read_index];
            host_range_min <= {{16{range_min[host_read_feature][15]}},
                               range_min[host_read_feature]};
            host_range_max <= {{16{range_max[host_read_feature][15]}},
                               range_max[host_read_feature]};
            shell_reference_count <= reference_hist[shell_read_index];
            shell_snapshot_count <= shell_selector_axis[16] ?
                snapshot_hist1[shell_read_index] : snapshot_hist0[shell_read_index];
            shell_range_min <= {{16{range_min[shell_read_feature][15]}},
                                range_min[shell_read_feature]};
            shell_range_max <= {{16{range_max[shell_read_feature][15]}},
                                range_max[shell_read_feature]};

            // The DNN accepts a packet every 672 cycles.  This monitor consumes
            // at most 320 cycles for a packet and 256 cycles for a completed
            // window snapshot, so it adds no backpressure.  Bin search and
            // snapshot transfer are serialized to keep the 250 MHz path short.
            if (sample_accept) begin
                if (processing_sample || snapshot_copy_active) begin
                    overflow_error <= 1'b1;
                end else begin
                    for (reset_index = 0; reset_index < FEATURE_COUNT; reset_index = reset_index + 1)
                        pending_features[reset_index*16 +: 16] <=
                            packet_feature_q8(sample_tdata, reset_index);
                    processing_sample <= 1'b1;
                    feature_cursor <= 0;
                    feature_state <= RANGE_PREPARE;
                    pending_window_last <=
                        window_sample_count == WINDOW_SIZE - 1 ||
                        total_sample_count == SAMPLE_COUNT - 1;
                    pending_window_samples <= window_sample_count + 1'b1;
                    total_sample_count <= total_sample_count + 1'b1;
                    if (window_sample_count == WINDOW_SIZE - 1 ||
                        total_sample_count == SAMPLE_COUNT - 1)
                        window_sample_count <= 0;
                    else
                        window_sample_count <= window_sample_count + 1'b1;
                end
            end

            if (snapshot_copy_active) begin
                for (reset_index = 0; reset_index < HIST_COUNT; reset_index = reset_index + 1) begin
                    if (snapshot_copy_enable[reset_index]) begin
                        if (!snapshot_copy_bank)
                            snapshot_hist0[reset_index] <= current_hist[reset_index];
                        else
                            snapshot_hist1[reset_index] <= current_hist[reset_index];
                        current_hist[reset_index] <= 0;
                    end
                end

                if (snapshot_copy_index == HIST_COUNT - 1) begin
                    snapshot_copy_active <= 1'b0;
                    snapshot_copy_index <= 0;
                    snapshot_copy_enable <= 0;
                    completed_window_id <= next_window_id;
                    completed_window_samples <= pending_window_samples;
                    snapshot_valid <= 1'b1;
                    next_window_id <= next_window_id + 1'b1;
                    if (next_window_id + 1'b1 == REFERENCE_WINDOWS)
                        reference_ready <= 1'b1;
                end else begin
                    snapshot_copy_index <= snapshot_copy_index + 1'b1;
                    snapshot_copy_enable <= snapshot_copy_enable << 1;
                end
            end else if (processing_sample) begin
                case (feature_state)
                    RANGE_PREPARE: begin
                        if (!reference_ready) begin
                            range_min[feature_cursor] <= candidate_minimum;
                            range_max[feature_cursor] <= candidate_maximum;
                            range_valid[feature_cursor] <= 1'b1;
                        end
                        prepared_minimum <= bin_minimum;
                        prepared_maximum <= bin_maximum;
                        feature_state <= BIN_PREPARE;
                    end

                    BIN_PREPARE: begin
                        if (prepared_maximum <= prepared_minimum ||
                            current_feature <= prepared_minimum) begin
                            selected_bin <= 0;
                            feature_state <= WRITE_SELECT;
                        end else if (current_feature >= prepared_maximum) begin
                            selected_bin <= BIN_COUNT - 1;
                            feature_state <= WRITE_SELECT;
                        end else begin
                            bin_candidate <= 0;
                            bin_delta_scaled <=
                                ($signed(current_feature) - $signed(prepared_minimum)) <<< 4;
                            bin_span <= $signed(prepared_maximum) - $signed(prepared_minimum);
                            bin_threshold <= $signed(prepared_maximum) - $signed(prepared_minimum);
                            feature_state <= BIN_SEARCH;
                        end
                    end

                    BIN_SEARCH: begin
                        if (bin_delta_scaled < bin_threshold || bin_candidate == BIN_COUNT - 1) begin
                            selected_bin <= bin_candidate;
                            feature_state <= WRITE_SELECT;
                        end else begin
                            bin_candidate <= bin_candidate + 1'b1;
                            bin_threshold <= bin_threshold + bin_span;
                        end
                    end

                    WRITE_SELECT: begin
                        histogram_write_enable <= 0;
                        histogram_write_enable[selected_histogram_index] <= 1'b1;
                        selected_reference_count <=
                            reference_hist[selected_histogram_index];
                        selected_current_count <=
                            current_hist[selected_histogram_index];
                        feature_state <= HISTOGRAM_UPDATE;
                    end

                    HISTOGRAM_UPDATE: begin
                        if (&selected_current_count ||
                            (!reference_ready && &selected_reference_count))
                            overflow_error <= 1'b1;
                        for (reset_index = 0; reset_index < HIST_COUNT; reset_index = reset_index + 1) begin
                            if (histogram_write_enable[reset_index]) begin
                                if (!reference_ready) begin
                                    reference_hist[reset_index] <=
                                        saturating_increment(selected_reference_count);
                                end
                                current_hist[reset_index] <=
                                    saturating_increment(selected_current_count);
                            end
                        end
                        histogram_write_enable <= 0;

                        if (feature_cursor == FEATURE_COUNT - 1) begin
                            processing_sample <= 1'b0;
                            if (pending_window_last) begin
                                snapshot_copy_active <= 1'b1;
                                snapshot_copy_bank <= next_window_id[0];
                                snapshot_copy_index <= 0;
                                snapshot_copy_enable <= {{(HIST_COUNT-1){1'b0}}, 1'b1};
                            end
                        end else begin
                            feature_cursor <= feature_cursor + 1'b1;
                            feature_state <= RANGE_PREPARE;
                        end
                    end

                    default: begin
                        feature_state <= RANGE_PREPARE;
                    end
                endcase
            end
        end
    end
endmodule
