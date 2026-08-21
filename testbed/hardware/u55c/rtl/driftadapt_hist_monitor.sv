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
    output wire [FEATURE_COUNT*BIN_COUNT*32-1:0] reference_hist_flat,
    output wire [FEATURE_COUNT*BIN_COUNT*32-1:0] snapshot_hist0_flat,
    output wire [FEATURE_COUNT*BIN_COUNT*32-1:0] snapshot_hist1_flat,
    output wire [FEATURE_COUNT*16-1:0]        range_min_flat,
    output wire [FEATURE_COUNT*16-1:0]        range_max_flat
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
    reg pending_window_last;
    reg [31:0] pending_window_samples;

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

    // Equal-width bin selection without a divider.  Cross multiplication maps
    // to fixed add/shift/compare logic for the constant sixteen-bin design.
    function automatic [3:0] histogram_bin(
        input signed [15:0] value,
        input signed [15:0] minimum,
        input signed [15:0] maximum
    );
        reg signed [31:0] delta;
        reg signed [31:0] span;
        integer candidate;
        begin
            histogram_bin = BIN_COUNT - 1;
            if (maximum <= minimum || value <= minimum) begin
                histogram_bin = 0;
            end else if (value >= maximum) begin
                histogram_bin = BIN_COUNT - 1;
            end else begin
                delta = value - minimum;
                span = maximum - minimum;
                for (candidate = 0; candidate < BIN_COUNT; candidate = candidate + 1)
                    if (delta * BIN_COUNT < span * (candidate + 1) &&
                        histogram_bin == BIN_COUNT - 1)
                        histogram_bin = candidate[3:0];
            end
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
    wire [3:0] selected_bin = histogram_bin(
        current_feature,
        reference_ready ? range_min[feature_cursor] : candidate_minimum,
        reference_ready ? range_max[feature_cursor] : candidate_maximum
    );
    wire [7:0] selected_histogram_index = feature_cursor * BIN_COUNT + selected_bin;

    genvar output_index;
    generate
        for (output_index = 0; output_index < HIST_COUNT; output_index = output_index + 1) begin : flatten_histograms
            assign reference_hist_flat[output_index*32 +: 32] = reference_hist[output_index];
            assign snapshot_hist0_flat[output_index*32 +: 32] = snapshot_hist0[output_index];
            assign snapshot_hist1_flat[output_index*32 +: 32] = snapshot_hist1[output_index];
        end
        for (output_index = 0; output_index < FEATURE_COUNT; output_index = output_index + 1) begin : flatten_ranges
            assign range_min_flat[output_index*16 +: 16] = range_min[output_index];
            assign range_max_flat[output_index*16 +: 16] = range_max[output_index];
        end
    endgenerate

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
            pending_window_last <= 1'b0;
            pending_window_samples <= 0;
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
            // The DNN accepts a packet every 672 cycles.  This monitor consumes
            // one of its sixteen features per cycle, so it adds no backpressure
            // and uses a single fixed bin-selector datapath.
            if (sample_accept) begin
                if (processing_sample) begin
                    overflow_error <= 1'b1;
                end else begin
                    for (reset_index = 0; reset_index < FEATURE_COUNT; reset_index = reset_index + 1)
                        pending_features[reset_index*16 +: 16] <=
                            packet_feature_q8(sample_tdata, reset_index);
                    processing_sample <= 1'b1;
                    feature_cursor <= 0;
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

            if (processing_sample) begin
                if (!reference_ready) begin
                    range_min[feature_cursor] <= candidate_minimum;
                    range_max[feature_cursor] <= candidate_maximum;
                    range_valid[feature_cursor] <= 1'b1;
                    if (&reference_hist[selected_histogram_index])
                        overflow_error <= 1'b1;
                    reference_hist[selected_histogram_index] <=
                        saturating_increment(reference_hist[selected_histogram_index]);
                end
                if (&current_hist[selected_histogram_index])
                    overflow_error <= 1'b1;

                if (pending_window_last && feature_cursor == FEATURE_COUNT - 1) begin
                    for (reset_index = 0; reset_index < HIST_COUNT; reset_index = reset_index + 1) begin
                        if (!next_window_id[0])
                            snapshot_hist0[reset_index] <=
                                (reset_index == selected_histogram_index) ?
                                saturating_increment(current_hist[reset_index]) : current_hist[reset_index];
                        else
                            snapshot_hist1[reset_index] <=
                                (reset_index == selected_histogram_index) ?
                                saturating_increment(current_hist[reset_index]) : current_hist[reset_index];
                        current_hist[reset_index] <= 0;
                    end
                end else begin
                    current_hist[selected_histogram_index] <=
                        saturating_increment(current_hist[selected_histogram_index]);
                end

                if (feature_cursor == FEATURE_COUNT - 1) begin
                    processing_sample <= 1'b0;
                    if (pending_window_last) begin
                        completed_window_id <= next_window_id;
                        completed_window_samples <= pending_window_samples;
                        snapshot_valid <= 1'b1;
                        next_window_id <= next_window_id + 1'b1;
                        if (next_window_id + 1'b1 == REFERENCE_WINDOWS)
                            reference_ready <= 1'b1;
                    end
                end else begin
                    feature_cursor <= feature_cursor + 1'b1;
                end
            end
        end
    end
endmodule
