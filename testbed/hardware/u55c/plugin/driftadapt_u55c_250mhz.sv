`timescale 1ns/1ps
`include "driftadapt_config.vh"
// Encoded MLP flow replay and verified-feedback model exchange, ABI 5.
module driftadapt_u55c_250mhz #(
    parameter integer NUM_QDMA = 1,
    parameter integer NUM_INTF = 2,
    parameter integer NUM_CMAC_PORT = 2,
    parameter integer SAMPLE_COUNT = `DA_SAMPLE_COUNT,
    parameter integer WINDOW_SIZE = `DA_WINDOW_SIZE,
    parameter integer STARTUP_CYCLES = 4096,
    parameter integer GAP_CYCLES = 1,
    parameter WEIGHT_MEM_FILE = "driftadapt_weights.mem",
    parameter SAMPLE_MEM_FILE = "driftadapt_samples.mem"
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

 wire axis_aresetn,axil_aresetn;
 generic_reset #(.NUM_INPUT_CLK(2),.RESET_DURATION(100)) reset_inst(
 .mod_rstn(mod_rstn),.mod_rst_done(mod_rst_done),.clk({axis_aclk,axil_aclk}),
 .rstn({axis_aresetn,axil_aresetn}));
 wire [511:0] generated,classified;
 wire generated_valid,generated_ready,dnn_ready,classified_valid,window_ready,hist_ready;
 wire finished,drift,loaded,updating,update_begin,stage_valid,commit;
 wire [8:0] staged_words,stage_index;
 wire [31:0] model_version,stage_data,hist_window,status;
 wire [2:0] errors;
 wire [63:0] sent;
 assign generated_ready=dnn_ready && hist_ready;
 driftadapt_packet_generator #(.SAMPLE_COUNT(SAMPLE_COUNT),.STARTUP_CYCLES(STARTUP_CYCLES),
 .GAP_CYCLES(GAP_CYCLES),.SAMPLE_MEM_FILE(SAMPLE_MEM_FILE)) source(
 .clk(axis_aclk),.rst_n(axis_aresetn),.m_axis_tdata(generated),
 .m_axis_tvalid(generated_valid),.m_axis_tready(generated_ready),
 .finished_axis(finished),.sent_packets_axis(sent));
 driftadapt_hist_monitor #(.WINDOW_SIZE(WINDOW_SIZE)) monitor(
 .clk(axis_aclk),.rst_n(axis_aresetn),.bin_ids(generated[112 +: 128]),
 .accept(generated_valid && generated_ready),.ready(hist_ready),.drift(drift),.window_id(hist_window));
 driftadapt_dnn_axis #(.WEIGHT_MEM_FILE(WEIGHT_MEM_FILE)) dnn(
 .clk(axis_aclk),.rst_n(axis_aresetn),.update_begin(update_begin),.stage_valid(stage_valid),
 .stage_index(stage_index),.stage_data(stage_data),.commit(commit),.loaded(loaded),
 .updating(updating),.staged_words(staged_words),.model_version(model_version),.errors(errors),
 .s_axis_tdata(generated),.s_axis_tvalid(generated_valid && hist_ready),.s_axis_tready(dnn_ready),
 .m_axis_tdata(classified),.m_axis_tvalid(classified_valid),.m_axis_tready(window_ready));
    wire [31:0] jtag_awaddr;
    wire [2:0] jtag_awprot;
    wire jtag_awvalid;
    wire jtag_awready;
    wire [31:0] jtag_wdata;
    wire [3:0] jtag_wstrb;
    wire jtag_wvalid;
    wire jtag_wready;
    wire [1:0] jtag_bresp;
    wire jtag_bvalid;
    wire jtag_bready;
    wire [31:0] jtag_araddr;
    wire [2:0] jtag_arprot;
    wire jtag_arvalid;
    wire jtag_arready;
    wire [31:0] jtag_rdata;
    wire [1:0] jtag_rresp;
    wire jtag_rvalid;
    wire jtag_rready;

    driftadapt_jtag_axi driftadapt_jtag_axi_master (
        .aclk(axis_aclk), .aresetn(axis_aresetn),
        .m_axi_awaddr(jtag_awaddr), .m_axi_awprot(jtag_awprot),
        .m_axi_awvalid(jtag_awvalid), .m_axi_awready(jtag_awready),
        .m_axi_wdata(jtag_wdata), .m_axi_wstrb(jtag_wstrb),
        .m_axi_wvalid(jtag_wvalid), .m_axi_wready(jtag_wready),
        .m_axi_bresp(jtag_bresp), .m_axi_bvalid(jtag_bvalid),
        .m_axi_bready(jtag_bready), .m_axi_araddr(jtag_araddr),
        .m_axi_arprot(jtag_arprot), .m_axi_arvalid(jtag_arvalid),
        .m_axi_arready(jtag_arready), .m_axi_rdata(jtag_rdata),
        .m_axi_rresp(jtag_rresp), .m_axi_rvalid(jtag_rvalid),
        .m_axi_rready(jtag_rready)
    );


 driftadapt_window_manager #(.WINDOW_SIZE(WINDOW_SIZE),.SAMPLE_COUNT(SAMPLE_COUNT)) windows(
 .clk(axis_aclk),.rst_n(axis_aresetn),.s_axis_tdata(classified),
 .s_axis_tvalid(classified_valid),.s_axis_tready(window_ready),
 .drift(drift),.loaded(loaded),.updating(updating),.staged_words(staged_words),
 .model_version(model_version),.model_errors(errors),.update_begin(update_begin),
 .stage_valid(stage_valid),.commit(commit),.stage_index(stage_index),.stage_data(stage_data),.status(status),
 .s_axil_awvalid(jtag_awvalid),.s_axil_awaddr(jtag_awaddr),.s_axil_awready(jtag_awready),
 .s_axil_wvalid(jtag_wvalid),.s_axil_wdata(jtag_wdata),.s_axil_wstrb(jtag_wstrb),
 .s_axil_wready(jtag_wready),.s_axil_bvalid(jtag_bvalid),.s_axil_bresp(jtag_bresp),.s_axil_bready(jtag_bready),
 .s_axil_arvalid(jtag_arvalid),.s_axil_araddr(jtag_araddr),.s_axil_arready(jtag_arready),
 .s_axil_rvalid(jtag_rvalid),.s_axil_rdata(jtag_rdata),.s_axil_rresp(jtag_rresp),.s_axil_rready(jtag_rready));
 driftadapt_control_regs public_regs(
 .clk(axil_aclk),.rst_n(axil_aresetn),
 .s_axil_awvalid(s_axil_awvalid),.s_axil_awaddr(s_axil_awaddr),.s_axil_awready(s_axil_awready),
 .s_axil_wvalid(s_axil_wvalid),.s_axil_wdata(s_axil_wdata),.s_axil_wready(s_axil_wready),
 .s_axil_bvalid(s_axil_bvalid),.s_axil_bresp(s_axil_bresp),.s_axil_bready(s_axil_bready),
 .s_axil_arvalid(s_axil_arvalid),.s_axil_araddr(s_axil_araddr),.s_axil_arready(s_axil_arready),
 .s_axil_rvalid(s_axil_rvalid),.s_axil_rdata(s_axil_rdata),.s_axil_rresp(s_axil_rresp),.s_axil_rready(s_axil_rready));
    assign m_axis_adap_tx_250mhz_tvalid = {NUM_CMAC_PORT{1'b0}};
    assign m_axis_adap_tx_250mhz_tdata = {512*NUM_CMAC_PORT{1'b0}};
    assign m_axis_adap_tx_250mhz_tkeep = {64*NUM_CMAC_PORT{1'b0}};
    assign m_axis_adap_tx_250mhz_tlast = {NUM_CMAC_PORT{1'b0}};
    assign m_axis_adap_tx_250mhz_tuser_size = {16*NUM_CMAC_PORT{1'b0}};
    assign m_axis_adap_tx_250mhz_tuser_src = {16*NUM_CMAC_PORT{1'b0}};
    assign m_axis_adap_tx_250mhz_tuser_dst = {16*NUM_CMAC_PORT{1'b0}};
    assign s_axis_adap_rx_250mhz_tready = {NUM_CMAC_PORT{1'b1}};

    assign s_axis_qdma_h2c_tready = {NUM_INTF*NUM_QDMA{1'b0}};
    assign m_axis_qdma_c2h_tvalid = {NUM_INTF*NUM_QDMA{1'b0}};
    assign m_axis_qdma_c2h_tdata = {512*NUM_INTF*NUM_QDMA{1'b0}};
    assign m_axis_qdma_c2h_tkeep = {64*NUM_INTF*NUM_QDMA{1'b0}};
    assign m_axis_qdma_c2h_tlast = {NUM_INTF*NUM_QDMA{1'b0}};
    assign m_axis_qdma_c2h_tuser_size = {16*NUM_INTF*NUM_QDMA{1'b0}};
    assign m_axis_qdma_c2h_tuser_src = {16*NUM_INTF*NUM_QDMA{1'b0}};
    assign m_axis_qdma_c2h_tuser_dst = {16*NUM_INTF*NUM_QDMA{1'b0}};

endmodule
