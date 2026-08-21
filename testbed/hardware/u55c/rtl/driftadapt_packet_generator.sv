`timescale 1ns/1ps

// Autonomous CIC-IDS2017 packet source. Each ROM word contains only sixteen
// signed Q8.8 features in bits [255:0]. Ground-truth labels are intentionally
// absent from the synthesized traffic image.
module driftadapt_packet_generator #(
    parameter integer SAMPLE_COUNT = 16000,
    parameter integer STARTUP_CYCLES = 1250000000,
    parameter integer GAP_CYCLES = 1024,
    parameter SAMPLE_MEM_FILE = "driftadapt_samples.mem"
) (
    input  wire          clk,
    input  wire          rst_n,
    output wire [511:0]  m_axis_tdata,
    output wire          m_axis_tvalid,
    input  wire          m_axis_tready,
    output wire          finished_axis,
    output reg  [63:0]   sent_packets_axis
);
    localparam integer SAMPLE_INDEX_WIDTH = $clog2(SAMPLE_COUNT);
    localparam integer STARTUP_WIDTH = $clog2(STARTUP_CYCLES + 1);
    localparam integer GAP_WIDTH = $clog2(GAP_CYCLES + 1);
    localparam [2:0] WAIT_STARTUP = 3'd0, LOAD_SAMPLE = 3'd1,
                     SEND_PACKET = 3'd2, WAIT_GAP = 3'd3, FINISHED = 3'd4;

    (* rom_style = "block" *) reg [255:0] sample_rom [0:SAMPLE_COUNT-1];
    reg [255:0] sample_word;
    reg [SAMPLE_INDEX_WIDTH-1:0] sample_index;
    reg [STARTUP_WIDTH-1:0] startup_count;
    reg [GAP_WIDTH-1:0] gap_count;
    reg [2:0] state;
    reg [511:0] packet_data;
    integer feature_index;

    initial begin
        if (SAMPLE_COUNT <= 0 || STARTUP_CYCLES <= 0 || GAP_CYCLES <= 0)
            $fatal(1, "DRIFTADAPT generator parameters must be positive");
        $readmemh(SAMPLE_MEM_FILE, sample_rom);
    end

    always @(*) begin
        packet_data = 512'd0;
        // Broadcast destination and a locally administered source address.
        packet_data[0*8 +: 8] = 8'hff;
        packet_data[1*8 +: 8] = 8'hff;
        packet_data[2*8 +: 8] = 8'hff;
        packet_data[3*8 +: 8] = 8'hff;
        packet_data[4*8 +: 8] = 8'hff;
        packet_data[5*8 +: 8] = 8'hff;
        packet_data[6*8 +: 8] = 8'h02;
        packet_data[7*8 +: 8] = 8'h00;
        packet_data[8*8 +: 8] = 8'h00;
        packet_data[9*8 +: 8] = 8'h00;
        packet_data[10*8 +: 8] = 8'h00;
        packet_data[11*8 +: 8] = 8'h01;
        packet_data[12*8 +: 8] = 8'h90;
        packet_data[13*8 +: 8] = 8'h00;
        for (feature_index = 0; feature_index < 16; feature_index = feature_index + 1) begin
            packet_data[(14 + feature_index*2)*8 +: 8] =
                sample_word[feature_index*16 + 8 +: 8];
            packet_data[(15 + feature_index*2)*8 +: 8] =
                sample_word[feature_index*16 +: 8];
        end
        // Byte 46 was used by the old in-FPGA oracle checker. Keep it zero so
        // the local labeling agent is the only online reference source.
        packet_data[46*8 +: 8] = 8'd0;
        packet_data[47*8 +: 8] = 8'd0;
        packet_data[48*8 +: 8] = 8'hff;
    end

    assign m_axis_tdata = packet_data;
    assign m_axis_tvalid = state == SEND_PACKET;
    assign finished_axis = state == FINISHED;

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= WAIT_STARTUP;
            startup_count <= 0;
            gap_count <= 0;
            sample_index <= 0;
            sample_word <= 0;
            sent_packets_axis <= 0;
        end else begin
            case (state)
                WAIT_STARTUP: begin
                    if (startup_count == STARTUP_CYCLES - 1) begin
                        startup_count <= 0;
                        state <= LOAD_SAMPLE;
                    end else begin
                        startup_count <= startup_count + 1'b1;
                    end
                end

                LOAD_SAMPLE: begin
                    sample_word <= sample_rom[sample_index];
                    state <= SEND_PACKET;
                end

                SEND_PACKET: begin
                    if (m_axis_tready) begin
                        sent_packets_axis <= sent_packets_axis + 1'b1;
                        if (sample_index == SAMPLE_COUNT - 1) begin
                            state <= FINISHED;
                        end else begin
                            sample_index <= sample_index + 1'b1;
                            gap_count <= 0;
                            state <= WAIT_GAP;
                        end
                    end
                end

                WAIT_GAP: begin
                    if (gap_count == GAP_CYCLES - 1) begin
                        gap_count <= 0;
                        state <= LOAD_SAMPLE;
                    end else begin
                        gap_count <= gap_count + 1'b1;
                    end
                end

                default: state <= FINISHED;
            endcase
        end
    end
endmodule
