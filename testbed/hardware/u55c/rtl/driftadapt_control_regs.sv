`timescale 1ns/1ps

// BAR2 user block at 0x100000. Software writes all 182 shadow parameters and
// then writes bit 0 at offset 0x008. The axis domain copies the stable bundled
// data only after the synchronized commit toggle changes.
module driftadapt_control_regs (
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

    output wire [5823:0] weight_shadow_axil,
    output reg           commit_toggle_axil,
    input  wire          commit_ack_axis,
    input  wire          weights_loaded_axis,
    input  wire [63:0]   classified_packets_axis,
    input  wire [63:0]   bypassed_packets_axis
);
    localparam [31:0] FEATURE_WORD = 32'h44524654; // "DRFT"
    localparam integer PARAM_COUNT = 182;
    localparam [11:0] WEIGHT_BASE = 12'h010;

    reg [31:0] weight_shadow [0:PARAM_COUNT-1];
    reg        aw_pending;
    reg [11:0] awaddr_pending;
    reg        w_pending;
    reg [31:0] wdata_pending;
    integer index;

    (* ASYNC_REG = "TRUE" *) reg commit_ack_meta, commit_ack_axil;
    (* ASYNC_REG = "TRUE" *) reg weights_loaded_meta, weights_loaded_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] classified_meta, classified_axil;
    (* ASYNC_REG = "TRUE" *) reg [63:0] bypassed_meta, bypassed_axil;

    wire commit_busy = commit_toggle_axil != commit_ack_axil;

    generate
        genvar parameter_index;
        for (parameter_index = 0; parameter_index < PARAM_COUNT; parameter_index = parameter_index + 1) begin : flatten_parameters
            assign weight_shadow_axil[parameter_index*32 +: 32] = weight_shadow[parameter_index];
        end
    endgenerate

    assign s_axil_awready = axil_aresetn && !aw_pending && !s_axil_bvalid;
    assign s_axil_wready  = axil_aresetn && !w_pending && !s_axil_bvalid;
    assign s_axil_bresp   = 2'b00;
    assign s_axil_arready = axil_aresetn && !s_axil_rvalid;
    assign s_axil_rresp   = 2'b00;

    function automatic [31:0] read_register(input [11:0] address);
        integer parameter_number;
        begin
            read_register = 32'd0;
            case (address)
                12'h000: read_register = FEATURE_WORD;
                12'h004: read_register = {30'd0, commit_busy, weights_loaded_axil};
                12'h008: read_register = {31'd0, commit_toggle_axil};
                12'h300: read_register = classified_axil[31:0];
                12'h304: read_register = classified_axil[63:32];
                12'h308: read_register = bypassed_axil[31:0];
                12'h30c: read_register = bypassed_axil[63:32];
                default: begin
                    if (address >= WEIGHT_BASE && address < WEIGHT_BASE + PARAM_COUNT*4 && address[1:0] == 2'b00) begin
                        parameter_number = (address - WEIGHT_BASE) >> 2;
                        read_register = weight_shadow[parameter_number];
                    end
                end
            endcase
        end
    endfunction

    always @(posedge axil_aclk) begin
        if (!axil_aresetn) begin
            aw_pending <= 1'b0;
            awaddr_pending <= 12'd0;
            w_pending <= 1'b0;
            wdata_pending <= 32'd0;
            s_axil_bvalid <= 1'b0;
            s_axil_rvalid <= 1'b0;
            s_axil_rdata <= 32'd0;
            commit_toggle_axil <= 1'b0;
            for (index = 0; index < PARAM_COUNT; index = index + 1)
                weight_shadow[index] <= 32'd0;
        end else begin
            if (s_axil_awvalid && s_axil_awready) begin
                aw_pending <= 1'b1;
                awaddr_pending <= s_axil_awaddr[11:0];
            end
            if (s_axil_wvalid && s_axil_wready) begin
                w_pending <= 1'b1;
                wdata_pending <= s_axil_wdata;
            end
            if (aw_pending && w_pending && !s_axil_bvalid) begin
                if (awaddr_pending == 12'h008 && wdata_pending[0]) begin
                    commit_toggle_axil <= ~commit_toggle_axil;
                end else if (awaddr_pending >= WEIGHT_BASE &&
                             awaddr_pending < WEIGHT_BASE + PARAM_COUNT*4 &&
                             awaddr_pending[1:0] == 2'b00) begin
                    weight_shadow[(awaddr_pending - WEIGHT_BASE) >> 2] <= wdata_pending;
                end
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
            commit_ack_meta <= 1'b0;
            commit_ack_axil <= 1'b0;
            weights_loaded_meta <= 1'b0;
            weights_loaded_axil <= 1'b0;
            classified_meta <= 64'd0;
            classified_axil <= 64'd0;
            bypassed_meta <= 64'd0;
            bypassed_axil <= 64'd0;
        end else begin
            commit_ack_meta <= commit_ack_axis;
            commit_ack_axil <= commit_ack_meta;
            weights_loaded_meta <= weights_loaded_axis;
            weights_loaded_axil <= weights_loaded_meta;
            classified_meta <= classified_packets_axis;
            classified_axil <= classified_meta;
            bypassed_meta <= bypassed_packets_axis;
            bypassed_axil <= bypassed_meta;
        end
    end
endmodule
