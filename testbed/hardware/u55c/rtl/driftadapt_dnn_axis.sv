`timescale 1ns/1ps

// Single-packet DRIFTADAPT dataplane for the 512-bit OpenNIC stream. The sender's
// 16 signed Q8.8 features occupy Ethernet bytes 14..45; label/output/valid are
// bytes 46/47/48. Parameters and internal activations use signed Q16.16.
//
// One deliberately pipelined 32-bit MAC is shared by all neurons. This costs
// 672 axis cycles per classified packet, but gives every multiplier and adder
// an explicit register boundary at the U55C's 250 MHz stream clock.
module driftadapt_dnn_axis (
    input  wire          clk,
    input  wire          rst_n,
    input  wire [5823:0] weight_shadow_axil,
    input  wire          commit_toggle_axil,
    output reg           commit_ack_axis,
    output reg           weights_loaded_axis,
    output reg  [63:0]   classified_packets_axis,
    output reg  [63:0]   bypassed_packets_axis,

    input  wire [511:0]  s_axis_tdata,
    input  wire [63:0]   s_axis_tkeep,
    input  wire          s_axis_tvalid,
    output wire          s_axis_tready,
    input  wire          s_axis_tlast,
    input  wire [15:0]   s_axis_tuser_size,

    output wire [511:0]  m_axis_tdata,
    output wire [63:0]   m_axis_tkeep,
    output wire          m_axis_tvalid,
    input  wire          m_axis_tready,
    output wire          m_axis_tlast,
    output wire [15:0]   m_axis_tuser_size
);
    localparam integer PARAM_COUNT = 182;
    localparam [3:0] IDLE = 4'd0, LOAD_OPERANDS = 4'd1,
                     MULTIPLY = 4'd2, SCALE_PRODUCT = 4'd3,
                     ACCUMULATE = 4'd4, OUTPUT_PACKET = 4'd5,
                     PASS_REMAINDER = 4'd6;

    reg [3:0] state;
    reg [1:0] layer_index;
    reg [3:0] neuron_index;
    reg [4:0] feature_index;
    reg signed [31:0] active_weight [0:PARAM_COUNT-1];
    reg signed [31:0] activation [0:15];
    reg signed [63:0] accumulator;
    reg signed [31:0] operand_weight;
    reg signed [31:0] operand_activation;
    reg signed [63:0] product;
    reg signed [63:0] scaled_product;
    reg signed [63:0] first_logit;
    reg [511:0] packet_data;
    reg [63:0] packet_keep;
    reg packet_last;
    reg [15:0] packet_size;
    integer index;

    (* ASYNC_REG = "TRUE" *) reg commit_meta, commit_sync;

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

    wire commit_pending = commit_sync != commit_ack_axis;
    wire valid_feature_packet = s_axis_tlast && (&s_axis_tkeep[48:14]) &&
                                s_axis_tdata[48*8 +: 8] == 8'hff;
    wire signed [63:0] accumulated_value = accumulator + scaled_product;

    assign s_axis_tready = state == PASS_REMAINDER ? m_axis_tready :
                           state == IDLE && weights_loaded_axis && !commit_pending;
    assign m_axis_tvalid = state == OUTPUT_PACKET ? 1'b1 :
                           state == PASS_REMAINDER ? s_axis_tvalid : 1'b0;
    assign m_axis_tdata = state == PASS_REMAINDER ? s_axis_tdata : packet_data;
    assign m_axis_tkeep = state == PASS_REMAINDER ? s_axis_tkeep : packet_keep;
    assign m_axis_tlast = state == PASS_REMAINDER ? s_axis_tlast : packet_last;
    assign m_axis_tuser_size = state == PASS_REMAINDER ? s_axis_tuser_size : packet_size;

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
            packet_keep <= 0;
            packet_last <= 0;
            packet_size <= 0;
            commit_meta <= 0;
            commit_sync <= 0;
            commit_ack_axis <= 0;
            weights_loaded_axis <= 0;
            classified_packets_axis <= 0;
            bypassed_packets_axis <= 0;
            for (index = 0; index < PARAM_COUNT; index = index + 1)
                active_weight[index] <= 0;
            for (index = 0; index < 16; index = index + 1)
                activation[index] <= 0;
        end else begin
            commit_meta <= commit_toggle_axil;
            commit_sync <= commit_meta;

            if (state == IDLE && commit_pending) begin
                for (index = 0; index < PARAM_COUNT; index = index + 1)
                    active_weight[index] <= weight_shadow_axil[index*32 +: 32];
                commit_ack_axis <= commit_sync;
                weights_loaded_axis <= 1'b1;
            end

            case (state)
                IDLE: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        packet_data <= s_axis_tdata;
                        packet_keep <= s_axis_tkeep;
                        packet_last <= s_axis_tlast;
                        packet_size <= s_axis_tuser_size;
                        if (valid_feature_packet) begin
                            for (index = 0; index < 16; index = index + 1)
                                activation[index] <= packet_feature_q16(s_axis_tdata, index);
                            layer_index <= 0;
                            neuron_index <= 0;
                            feature_index <= 0;
                            accumulator <= {{32{active_weight[128][31]}}, active_weight[128]};
                            state <= LOAD_OPERANDS;
                        end else begin
                            bypassed_packets_axis <= bypassed_packets_axis + 1'b1;
                            state <= OUTPUT_PACKET;
                        end
                    end
                end

                LOAD_OPERANDS: begin
                    operand_activation <= activation[feature_index];
                    case (layer_index)
                        0: operand_weight <= active_weight[neuron_index*16 + feature_index];
                        1: operand_weight <= active_weight[136 + neuron_index*8 + feature_index];
                        default: operand_weight <= active_weight[172 + neuron_index*4 + feature_index];
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
                                    accumulator <= {{32{active_weight[168][31]}}, active_weight[168]};
                                end else begin
                                    neuron_index <= neuron_index + 1'b1;
                                    accumulator <= {{32{active_weight[129+neuron_index][31]}},
                                                    active_weight[129+neuron_index]};
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
                                    accumulator <= {{32{active_weight[180][31]}}, active_weight[180]};
                                end else begin
                                    neuron_index <= neuron_index + 1'b1;
                                    accumulator <= {{32{active_weight[169+neuron_index][31]}},
                                                    active_weight[169+neuron_index]};
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
                                    accumulator <= {{32{active_weight[181][31]}}, active_weight[181]};
                                    state <= LOAD_OPERANDS;
                                end else begin
                                    packet_data[47*8 +: 8] <=
                                        first_logit >= accumulated_value ? 8'd0 : 8'd1;
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
                        state <= packet_last ? IDLE : PASS_REMAINDER;
                end

                PASS_REMAINDER: begin
                    if (s_axis_tvalid && m_axis_tready && s_axis_tlast)
                        state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
