`timescale 1ns/1ps

module driftadapt_u55c_250mhz #(
    parameter integer NUM_QDMA = 1,
    parameter integer NUM_INTF = 2,
    parameter integer NUM_CMAC_PORT = 2
) (
    input wire s_axil_awvalid, input wire [31:0] s_axil_awaddr,
    output wire s_axil_awready, input wire s_axil_wvalid,
    input wire [31:0] s_axil_wdata, output wire s_axil_wready,
    output wire s_axil_bvalid, output wire [1:0] s_axil_bresp,
    input wire s_axil_bready, input wire s_axil_arvalid,
    input wire [31:0] s_axil_araddr, output wire s_axil_arready,
    output wire s_axil_rvalid, output wire [31:0] s_axil_rdata,
    output wire [1:0] s_axil_rresp, input wire s_axil_rready,

    input wire [NUM_INTF*NUM_QDMA-1:0] s_axis_qdma_h2c_tvalid,
    input wire [512*NUM_INTF*NUM_QDMA-1:0] s_axis_qdma_h2c_tdata,
    input wire [64*NUM_INTF*NUM_QDMA-1:0] s_axis_qdma_h2c_tkeep,
    input wire [NUM_INTF*NUM_QDMA-1:0] s_axis_qdma_h2c_tlast,
    input wire [16*NUM_INTF*NUM_QDMA-1:0] s_axis_qdma_h2c_tuser_size,
    input wire [16*NUM_INTF*NUM_QDMA-1:0] s_axis_qdma_h2c_tuser_src,
    input wire [16*NUM_INTF*NUM_QDMA-1:0] s_axis_qdma_h2c_tuser_dst,
    output wire [NUM_INTF*NUM_QDMA-1:0] s_axis_qdma_h2c_tready,

    output wire [NUM_INTF*NUM_QDMA-1:0] m_axis_qdma_c2h_tvalid,
    output wire [512*NUM_INTF*NUM_QDMA-1:0] m_axis_qdma_c2h_tdata,
    output wire [64*NUM_INTF*NUM_QDMA-1:0] m_axis_qdma_c2h_tkeep,
    output wire [NUM_INTF*NUM_QDMA-1:0] m_axis_qdma_c2h_tlast,
    output wire [16*NUM_INTF*NUM_QDMA-1:0] m_axis_qdma_c2h_tuser_size,
    output wire [16*NUM_INTF*NUM_QDMA-1:0] m_axis_qdma_c2h_tuser_src,
    output wire [16*NUM_INTF*NUM_QDMA-1:0] m_axis_qdma_c2h_tuser_dst,
    input wire [NUM_INTF*NUM_QDMA-1:0] m_axis_qdma_c2h_tready,

    output wire [NUM_CMAC_PORT-1:0] m_axis_adap_tx_250mhz_tvalid,
    output wire [512*NUM_CMAC_PORT-1:0] m_axis_adap_tx_250mhz_tdata,
    output wire [64*NUM_CMAC_PORT-1:0] m_axis_adap_tx_250mhz_tkeep,
    output wire [NUM_CMAC_PORT-1:0] m_axis_adap_tx_250mhz_tlast,
    output wire [16*NUM_CMAC_PORT-1:0] m_axis_adap_tx_250mhz_tuser_size,
    output wire [16*NUM_CMAC_PORT-1:0] m_axis_adap_tx_250mhz_tuser_src,
    output wire [16*NUM_CMAC_PORT-1:0] m_axis_adap_tx_250mhz_tuser_dst,
    input wire [NUM_CMAC_PORT-1:0] m_axis_adap_tx_250mhz_tready,

    input wire [NUM_CMAC_PORT-1:0] s_axis_adap_rx_250mhz_tvalid,
    input wire [512*NUM_CMAC_PORT-1:0] s_axis_adap_rx_250mhz_tdata,
    input wire [64*NUM_CMAC_PORT-1:0] s_axis_adap_rx_250mhz_tkeep,
    input wire [NUM_CMAC_PORT-1:0] s_axis_adap_rx_250mhz_tlast,
    input wire [16*NUM_CMAC_PORT-1:0] s_axis_adap_rx_250mhz_tuser_size,
    input wire [16*NUM_CMAC_PORT-1:0] s_axis_adap_rx_250mhz_tuser_src,
    input wire [16*NUM_CMAC_PORT-1:0] s_axis_adap_rx_250mhz_tuser_dst,
    output wire [NUM_CMAC_PORT-1:0] s_axis_adap_rx_250mhz_tready,

    input wire mod_rstn, output wire mod_rst_done,
    input wire axil_aclk, input wire axis_aclk
);
    initial begin
        if (NUM_QDMA != 1 || NUM_INTF != 2 || NUM_CMAC_PORT != 2)
            $fatal(1, "DRIFTADAPT U55C requires one QDMA, two PFs, and two CMAC ports");
    end

    wire axil_aresetn;
    wire axis_aresetn;
    generic_reset #(.NUM_INPUT_CLK(2), .RESET_DURATION(100)) reset_inst (
        .mod_rstn(mod_rstn), .mod_rst_done(mod_rst_done),
        .clk({axis_aclk, axil_aclk}), .rstn({axis_aresetn, axil_aresetn})
    );

    wire [5823:0] weight_shadow;
    wire commit_toggle;
    wire commit_ack;
    wire weights_loaded;
    wire [63:0] classified_packets;
    wire [63:0] bypassed_packets;

    driftadapt_control_regs control_regs (
        .s_axil_awvalid(s_axil_awvalid), .s_axil_awaddr(s_axil_awaddr),
        .s_axil_awready(s_axil_awready), .s_axil_wvalid(s_axil_wvalid),
        .s_axil_wdata(s_axil_wdata), .s_axil_wready(s_axil_wready),
        .s_axil_bvalid(s_axil_bvalid), .s_axil_bresp(s_axil_bresp),
        .s_axil_bready(s_axil_bready), .s_axil_arvalid(s_axil_arvalid),
        .s_axil_araddr(s_axil_araddr), .s_axil_arready(s_axil_arready),
        .s_axil_rvalid(s_axil_rvalid), .s_axil_rdata(s_axil_rdata),
        .s_axil_rresp(s_axil_rresp), .s_axil_rready(s_axil_rready),
        .axil_aclk(axil_aclk), .axil_aresetn(axil_aresetn),
        .weight_shadow_axil(weight_shadow), .commit_toggle_axil(commit_toggle),
        .commit_ack_axis(commit_ack), .weights_loaded_axis(weights_loaded),
        .classified_packets_axis(classified_packets),
        .bypassed_packets_axis(bypassed_packets)
    );

    wire [511:0] dnn_tdata;
    wire [63:0] dnn_tkeep;
    wire dnn_tvalid;
    wire dnn_tlast;
    wire [15:0] dnn_tuser_size;

    driftadapt_dnn_axis dnn (
        .clk(axis_aclk), .rst_n(axis_aresetn),
        .weight_shadow_axil(weight_shadow), .commit_toggle_axil(commit_toggle),
        .commit_ack_axis(commit_ack), .weights_loaded_axis(weights_loaded),
        .classified_packets_axis(classified_packets),
        .bypassed_packets_axis(bypassed_packets),
        .s_axis_tdata(s_axis_qdma_h2c_tdata[511:0]),
        .s_axis_tkeep(s_axis_qdma_h2c_tkeep[63:0]),
        .s_axis_tvalid(s_axis_qdma_h2c_tvalid[0]),
        .s_axis_tready(s_axis_qdma_h2c_tready[0]),
        .s_axis_tlast(s_axis_qdma_h2c_tlast[0]),
        .s_axis_tuser_size(s_axis_qdma_h2c_tuser_size[15:0]),
        .m_axis_tdata(dnn_tdata), .m_axis_tkeep(dnn_tkeep),
        .m_axis_tvalid(dnn_tvalid),
        .m_axis_tready(m_axis_adap_tx_250mhz_tready[0]),
        .m_axis_tlast(dnn_tlast), .m_axis_tuser_size(dnn_tuser_size)
    );

    // PF0 host TX is classified and emitted on QSFP0.
    assign m_axis_adap_tx_250mhz_tvalid[0] = dnn_tvalid;
    assign m_axis_adap_tx_250mhz_tdata[511:0] = dnn_tdata;
    assign m_axis_adap_tx_250mhz_tkeep[63:0] = dnn_tkeep;
    assign m_axis_adap_tx_250mhz_tlast[0] = dnn_tlast;
    assign m_axis_adap_tx_250mhz_tuser_size[15:0] = dnn_tuser_size;
    assign m_axis_adap_tx_250mhz_tuser_src[15:0] = s_axis_qdma_h2c_tuser_src[15:0];
    assign m_axis_adap_tx_250mhz_tuser_dst[15:0] = 16'h0040;

    // PF1 host TX remains a direct diagnostic path to QSFP1.
    assign m_axis_adap_tx_250mhz_tvalid[1] = s_axis_qdma_h2c_tvalid[1];
    assign m_axis_adap_tx_250mhz_tdata[1023:512] = s_axis_qdma_h2c_tdata[1023:512];
    assign m_axis_adap_tx_250mhz_tkeep[127:64] = s_axis_qdma_h2c_tkeep[127:64];
    assign m_axis_adap_tx_250mhz_tlast[1] = s_axis_qdma_h2c_tlast[1];
    assign m_axis_adap_tx_250mhz_tuser_size[31:16] = s_axis_qdma_h2c_tuser_size[31:16];
    assign m_axis_adap_tx_250mhz_tuser_src[31:16] = s_axis_qdma_h2c_tuser_src[31:16];
    assign m_axis_adap_tx_250mhz_tuser_dst[31:16] = 16'h0080;
    assign s_axis_qdma_h2c_tready[1] = m_axis_adap_tx_250mhz_tready[1];

    // Both receive ports return directly to their matching host PF. The
    // expected experiment path is QSFP0 TX -> external forwarder -> QSFP1 RX.
    assign m_axis_qdma_c2h_tvalid = s_axis_adap_rx_250mhz_tvalid;
    assign m_axis_qdma_c2h_tdata = s_axis_adap_rx_250mhz_tdata;
    assign m_axis_qdma_c2h_tkeep = s_axis_adap_rx_250mhz_tkeep;
    assign m_axis_qdma_c2h_tlast = s_axis_adap_rx_250mhz_tlast;
    assign m_axis_qdma_c2h_tuser_size = s_axis_adap_rx_250mhz_tuser_size;
    assign m_axis_qdma_c2h_tuser_src = s_axis_adap_rx_250mhz_tuser_src;
    assign m_axis_qdma_c2h_tuser_dst[15:0] = 16'h0001;
    assign m_axis_qdma_c2h_tuser_dst[31:16] = 16'h0002;
    assign s_axis_adap_rx_250mhz_tready = m_axis_qdma_c2h_tready;

    wire unused_h2c_dst = ^s_axis_qdma_h2c_tuser_dst;
    wire unused_rx_dst = ^s_axis_adap_rx_250mhz_tuser_dst;
endmodule
