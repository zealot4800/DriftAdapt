`timescale 1ns/1ps

// Cycle-accurate measurements for the internal, single-inference-at-a-time
// DRIFTADAPT datapath. The counters deliberately observe AXI-stream transfers,
// not valid alone, so backpressure is included in the reported performance.
module driftadapt_benchmark_metrics #(
    parameter integer SAMPLE_COUNT = 16000
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          input_valid,
    input  wire          input_ready,
    input  wire          output_valid,
    input  wire          output_ready,
    output wire          benchmark_active_axis,
    output reg           benchmark_complete_axis,
    output reg  [63:0]   total_cycles_axis,
    output reg  [63:0]   first_input_cycle_axis,
    output reg  [63:0]   last_output_cycle_axis,
    output reg  [63:0]   elapsed_cycles_axis,
    output reg  [63:0]   input_stall_cycles_axis,
    output reg  [63:0]   output_stall_cycles_axis,
    output reg  [63:0]   classifier_busy_cycles_axis,
    output reg  [63:0]   latency_sum_cycles_axis,
    output reg  [63:0]   latency_min_cycles_axis,
    output reg  [63:0]   latency_max_cycles_axis
);
    wire input_fire = input_valid && input_ready;
    wire output_fire = output_valid && output_ready;

    reg benchmark_started;
    reg packet_in_flight;
    reg [63:0] packet_start_cycle;
    reg [63:0] completed_packets;
    wire [63:0] current_latency =
        total_cycles_axis - packet_start_cycle + 64'd1;

    assign benchmark_active_axis = benchmark_started &&
                                   !benchmark_complete_axis;

    initial begin
        if (SAMPLE_COUNT <= 0)
            $fatal(1, "DRIFTADAPT metrics SAMPLE_COUNT must be positive");
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            benchmark_started <= 1'b0;
            benchmark_complete_axis <= 1'b0;
            packet_in_flight <= 1'b0;
            packet_start_cycle <= 64'd0;
            completed_packets <= 64'd0;
            total_cycles_axis <= 64'd0;
            first_input_cycle_axis <= 64'd0;
            last_output_cycle_axis <= 64'd0;
            elapsed_cycles_axis <= 64'd0;
            input_stall_cycles_axis <= 64'd0;
            output_stall_cycles_axis <= 64'd0;
            classifier_busy_cycles_axis <= 64'd0;
            latency_sum_cycles_axis <= 64'd0;
            latency_min_cycles_axis <= 64'd0;
            latency_max_cycles_axis <= 64'd0;
        end else begin
            if (!benchmark_complete_axis)
                total_cycles_axis <= total_cycles_axis + 64'd1;

            if (input_valid && !input_ready)
                input_stall_cycles_axis <= input_stall_cycles_axis + 64'd1;
            if (output_valid && !output_ready)
                output_stall_cycles_axis <= output_stall_cycles_axis + 64'd1;
            if (packet_in_flight || input_fire)
                classifier_busy_cycles_axis <= classifier_busy_cycles_axis + 64'd1;

            if (input_fire) begin
                packet_start_cycle <= total_cycles_axis;
                packet_in_flight <= 1'b1;
                if (!benchmark_started) begin
                    benchmark_started <= 1'b1;
                    first_input_cycle_axis <= total_cycles_axis;
                end
            end

            if (output_fire) begin
                packet_in_flight <= 1'b0;
                completed_packets <= completed_packets + 64'd1;
                latency_sum_cycles_axis <= latency_sum_cycles_axis + current_latency;
                if (completed_packets == 0 || current_latency < latency_min_cycles_axis)
                    latency_min_cycles_axis <= current_latency;
                if (completed_packets == 0 || current_latency > latency_max_cycles_axis)
                    latency_max_cycles_axis <= current_latency;

                if (completed_packets == SAMPLE_COUNT - 1) begin
                    benchmark_complete_axis <= 1'b1;
                    last_output_cycle_axis <= total_cycles_axis;
                    elapsed_cycles_axis <=
                        total_cycles_axis - first_input_cycle_axis + 64'd1;
                end
            end
        end
    end
endmodule
