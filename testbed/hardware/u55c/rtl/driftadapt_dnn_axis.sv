`timescale 1ns/1ps

// Single-packet DRIFTADAPT dataplane. The on-chip source's sixteen signed Q8.8
// features occupy Ethernet bytes 14..45; prediction/valid are bytes 47/48.
// Reserved byte 46 is deliberately ignored. Parameters and
// internal activations use signed Q16.16.
//
// One deliberately pipelined 32-bit MAC is shared by all neurons. This costs
// 672 axis cycles per classified packet, but gives every multiplier and adder
// an explicit register boundary at the U55C's 250 MHz stream clock.
module driftadapt_dnn_axis #(
    parameter WEIGHT_MEM_FILE = "driftadapt_weights.mem"
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          weight_stage_valid_axis,
    input  wire [7:0]    weight_stage_index_axis,
    input  wire [31:0]   weight_stage_data_axis,
    input  wire          weight_clone_request_axis,
    input  wire          selective_update_mode_axis,
    input  wire          weight_commit_request_axis,
    output reg           weight_clone_ack_axis,
    output reg           weight_clone_busy_axis,
    output reg           weight_commit_ack_axis,
    output reg           selective_verify_error_axis,
    output reg           weights_loaded_axis,
    output reg           active_weight_bank_axis,
    output reg  [31:0]   model_version_axis,
    output reg  [63:0]   classified_packets_axis,

    input  wire [511:0]  s_axis_tdata,
    input  wire          s_axis_tvalid,
    output wire          s_axis_tready,

    output wire [511:0]  m_axis_tdata,
    output wire          m_axis_tvalid,
    input  wire          m_axis_tready
);
    localparam integer PARAM_COUNT = 182;
    localparam [3:0] IDLE = 4'd0, LOAD_OPERANDS = 4'd1,
                     MULTIPLY = 4'd2, SCALE_PRODUCT = 4'd3,
                     ACCUMULATE = 4'd4, OUTPUT_PACKET = 4'd5;

    reg [3:0] state;
    reg [1:0] layer_index;
    reg [3:0] neuron_index;
    reg [4:0] feature_index;
    // Runtime updates are written only to the inactive bank. A commit swaps
    // banks at an inference boundary, so a packet never observes mixed weights.
    (* ram_style = "distributed" *) reg signed [31:0] weight_bank0 [0:PARAM_COUNT-1];
    (* ram_style = "distributed" *) reg signed [31:0] weight_bank1 [0:PARAM_COUNT-1];
    reg signed [31:0] activation [0:15];
    reg signed [63:0] accumulator;
    reg signed [31:0] operand_weight;
    reg signed [31:0] operand_activation;
    reg signed [63:0] product;
    reg signed [63:0] scaled_product;
    reg signed [63:0] first_logit;
    reg [511:0] packet_data;
    reg [7:0] clone_index;
    reg [7:0] verify_index;
    reg verify_busy;
    reg [PARAM_COUNT-1:0] selective_patch_mask;
    reg verify_mismatch;
    integer index;

    initial begin
        $readmemh(WEIGHT_MEM_FILE, weight_bank0);
        $readmemh(WEIGHT_MEM_FILE, weight_bank1);
    end

    function automatic signed [31:0] active_weight(input integer parameter_index);
        begin
            active_weight = active_weight_bank_axis ?
                            weight_bank1[parameter_index] :
                            weight_bank0[parameter_index];
        end
    endfunction

    function automatic signed [63:0] active_bias(input integer parameter_index);
        reg signed [31:0] selected_weight;
        begin
            selected_weight = active_weight_bank_axis ?
                              weight_bank1[parameter_index] :
                              weight_bank0[parameter_index];
            active_bias = {{32{selected_weight[31]}}, selected_weight};
        end
    endfunction

    function automatic signed [31:0] packet_feature_q16(
        input [511:0] data,
        input integer feature_number
    );
        reg [15:0] network_q8;
        reg signed [31:0] extended_q8;
        begin
            network_q8 = {
                data[(14 + feature_number*2)*8 +: 8],
                data[(15 + feature_number*2)*8 +: 8]
            };
            extended_q8 = {{16{network_q8[15]}}, network_q8};
            packet_feature_q16 = extended_q8 <<< 8;
        end
    endfunction

    function automatic signed [31:0] saturate_q16(input signed [63:0] value);
        begin
            if (value > 64'sh000000007fffffff)
                saturate_q16 = 32'sh7fffffff;
            else if (value < -64'sh0000000080000000)
                saturate_q16 = -32'sh80000000;
            else
                saturate_q16 = value[31:0];
        end
    endfunction

    function automatic signed [31:0] relu_q16(input signed [63:0] value);
        begin
            relu_q16 = value <= 0 ? 32'sd0 : saturate_q16(value);
        end
    endfunction

    wire signed [63:0] accumulated_value = accumulator + scaled_product;

    // Do not begin an inference unless the window recorder can eventually
    // accept it. This keeps per-inference latency separate from agent stalls.
    assign s_axis_tready = state == IDLE && !weight_clone_request_axis &&
                           !weight_clone_busy_axis && !weight_commit_request_axis &&
                           !verify_busy &&
                           m_axis_tready;
    assign m_axis_tvalid = state == OUTPUT_PACKET;
    assign m_axis_tdata = packet_data;

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            layer_index <= 0;
            neuron_index <= 0;
            feature_index <= 0;
            accumulator <= 0;
            operand_weight <= 0;
            operand_activation <= 0;
            product <= 0;
            scaled_product <= 0;
            first_logit <= 0;
            packet_data <= 0;
            clone_index <= 0;
            verify_index <= 0;
            verify_busy <= 1'b0;
            selective_patch_mask <= 0;
            verify_mismatch <= 1'b0;
            weight_clone_ack_axis <= 1'b0;
            weight_clone_busy_axis <= 1'b0;
            weight_commit_ack_axis <= 1'b0;
            selective_verify_error_axis <= 1'b0;
            weights_loaded_axis <= 1'b1;
            active_weight_bank_axis <= 1'b0;
            model_version_axis <= 32'd0;
            classified_packets_axis <= 0;
            for (index = 0; index < 16; index = index + 1)
                activation[index] <= 0;
        end else begin
            if (!weight_commit_request_axis)
                weight_commit_ack_axis <= 1'b0;
            if (!weight_clone_request_axis)
                weight_clone_ack_axis <= 1'b0;

            if (weight_stage_valid_axis && weight_stage_index_axis < PARAM_COUNT &&
                !weight_clone_busy_axis && !verify_busy) begin
                if (active_weight_bank_axis)
                    weight_bank0[weight_stage_index_axis] <= weight_stage_data_axis;
                else
                    weight_bank1[weight_stage_index_axis] <= weight_stage_data_axis;
                if (selective_update_mode_axis)
                    selective_patch_mask[weight_stage_index_axis] <= 1'b1;
            end

            if (state == IDLE && weight_clone_request_axis &&
                !weight_clone_ack_axis && !weight_clone_busy_axis && !verify_busy) begin
                weight_clone_busy_axis <= 1'b1;
                clone_index <= 0;
                selective_patch_mask <= 0;
                selective_verify_error_axis <= 1'b0;
            end else if (weight_clone_busy_axis) begin
                if (active_weight_bank_axis)
                    weight_bank0[clone_index] <= weight_bank1[clone_index];
                else
                    weight_bank1[clone_index] <= weight_bank0[clone_index];
                if (clone_index == PARAM_COUNT - 1) begin
                    weight_clone_busy_axis <= 1'b0;
                    weight_clone_ack_axis <= 1'b1;
                end else begin
                    clone_index <= clone_index + 1'b1;
                end
            end

            if (state == IDLE && weight_commit_request_axis &&
                !weight_commit_ack_axis && !weight_clone_busy_axis && !verify_busy) begin
                if (selective_update_mode_axis) begin
                    verify_busy <= 1'b1;
                    verify_index <= 0;
                    verify_mismatch <= 1'b0;
                    selective_verify_error_axis <= 1'b0;
                end else begin
                    selective_verify_error_axis <= 1'b0;
                    active_weight_bank_axis <= ~active_weight_bank_axis;
                    model_version_axis <= model_version_axis + 1'b1;
                    weight_commit_ack_axis <= 1'b1;
                    weights_loaded_axis <= 1'b1;
                end
            end else if (verify_busy) begin
                if (!selective_patch_mask[verify_index] &&
                    (active_weight_bank_axis ?
                     weight_bank0[verify_index] != weight_bank1[verify_index] :
                     weight_bank1[verify_index] != weight_bank0[verify_index]))
                    verify_mismatch <= 1'b1;

                if (verify_index == PARAM_COUNT - 1) begin
                    verify_busy <= 1'b0;
                    weight_commit_ack_axis <= 1'b1;
                    if (verify_mismatch ||
                        (!selective_patch_mask[verify_index] &&
                         (active_weight_bank_axis ?
                          weight_bank0[verify_index] != weight_bank1[verify_index] :
                          weight_bank1[verify_index] != weight_bank0[verify_index]))) begin
                        selective_verify_error_axis <= 1'b1;
                    end else begin
                        active_weight_bank_axis <= ~active_weight_bank_axis;
                        model_version_axis <= model_version_axis + 1'b1;
                        weights_loaded_axis <= 1'b1;
                    end
                end else begin
                    verify_index <= verify_index + 1'b1;
                end
            end

            case (state)
                IDLE: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        packet_data <= s_axis_tdata;
                        for (index = 0; index < 16; index = index + 1)
                            activation[index] <= packet_feature_q16(s_axis_tdata, index);
                        layer_index <= 0;
                        neuron_index <= 0;
                        feature_index <= 0;
                        accumulator <= active_bias(128);
                        state <= LOAD_OPERANDS;
                    end
                end

                LOAD_OPERANDS: begin
                    operand_activation <= activation[feature_index];
                    case (layer_index)
                        0: operand_weight <= active_weight(neuron_index*16 + feature_index);
                        1: operand_weight <= active_weight(136 + neuron_index*8 + feature_index);
                        default: operand_weight <= active_weight(172 + neuron_index*4 + feature_index);
                    endcase
                    state <= MULTIPLY;
                end

                MULTIPLY: begin
                    product <= operand_weight * operand_activation;
                    state <= SCALE_PRODUCT;
                end

                SCALE_PRODUCT: begin
                    scaled_product <= product >>> 16;
                    state <= ACCUMULATE;
                end

                ACCUMULATE: begin
                    case (layer_index)
                        0: begin
                            if (feature_index == 15) begin
                                activation[neuron_index] <= relu_q16(accumulated_value);
                                feature_index <= 0;
                                if (neuron_index == 7) begin
                                    layer_index <= 1;
                                    neuron_index <= 0;
                                    accumulator <= active_bias(168);
                                end else begin
                                    neuron_index <= neuron_index + 1'b1;
                                    accumulator <= active_bias(129+neuron_index);
                                end
                            end else begin
                                feature_index <= feature_index + 1'b1;
                                accumulator <= accumulated_value;
                            end
                            state <= LOAD_OPERANDS;
                        end

                        1: begin
                            if (feature_index == 7) begin
                                activation[neuron_index] <= relu_q16(accumulated_value);
                                feature_index <= 0;
                                if (neuron_index == 3) begin
                                    layer_index <= 2;
                                    neuron_index <= 0;
                                    accumulator <= active_bias(180);
                                end else begin
                                    neuron_index <= neuron_index + 1'b1;
                                    accumulator <= active_bias(169+neuron_index);
                                end
                            end else begin
                                feature_index <= feature_index + 1'b1;
                                accumulator <= accumulated_value;
                            end
                            state <= LOAD_OPERANDS;
                        end

                        default: begin
                            if (feature_index == 3) begin
                                feature_index <= 0;
                                if (neuron_index == 0) begin
                                    first_logit <= accumulated_value;
                                    neuron_index <= 1;
                                    accumulator <= active_bias(181);
                                    state <= LOAD_OPERANDS;
                                end else begin
                                    packet_data[47*8 +: 8] <=
                                        first_logit >= accumulated_value ? 8'd0 : 8'd1;
                                    packet_data[49*8 +: 32] <= saturate_q16(
                                        first_logit >= accumulated_value ?
                                        first_logit - accumulated_value :
                                        accumulated_value - first_logit
                                    );
                                    classified_packets_axis <= classified_packets_axis + 1'b1;
                                    state <= OUTPUT_PACKET;
                                end
                            end else begin
                                feature_index <= feature_index + 1'b1;
                                accumulator <= accumulated_value;
                                state <= LOAD_OPERANDS;
                            end
                        end
                    endcase
                end

                OUTPUT_PACKET: begin
                    if (m_axis_tready)
                        state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
