`timescale 1ns/1ps

module box_250mhz_address_map (
    input wire s_axil_awvalid, input wire [31:0] s_axil_awaddr,
    output wire s_axil_awready, input wire s_axil_wvalid,
    input wire [31:0] s_axil_wdata, output wire s_axil_wready,
    output wire s_axil_bvalid, output wire [1:0] s_axil_bresp,
    input wire s_axil_bready, input wire s_axil_arvalid,
    input wire [31:0] s_axil_araddr, output wire s_axil_arready,
    output wire s_axil_rvalid, output wire [31:0] s_axil_rdata,
    output wire [1:0] s_axil_rresp, input wire s_axil_rready,
    output wire m_axil_driftadapt_awvalid, output wire [31:0] m_axil_driftadapt_awaddr,
    input wire m_axil_driftadapt_awready, output wire m_axil_driftadapt_wvalid,
    output wire [31:0] m_axil_driftadapt_wdata, input wire m_axil_driftadapt_wready,
    input wire m_axil_driftadapt_bvalid, input wire [1:0] m_axil_driftadapt_bresp,
    output wire m_axil_driftadapt_bready, output wire m_axil_driftadapt_arvalid,
    output wire [31:0] m_axil_driftadapt_araddr, input wire m_axil_driftadapt_arready,
    input wire m_axil_driftadapt_rvalid, input wire [31:0] m_axil_driftadapt_rdata,
    input wire [1:0] m_axil_driftadapt_rresp, output wire m_axil_driftadapt_rready
);
    assign m_axil_driftadapt_awvalid = s_axil_awvalid;
    assign m_axil_driftadapt_awaddr = s_axil_awaddr;
    assign s_axil_awready = m_axil_driftadapt_awready;
    assign m_axil_driftadapt_wvalid = s_axil_wvalid;
    assign m_axil_driftadapt_wdata = s_axil_wdata;
    assign s_axil_wready = m_axil_driftadapt_wready;
    assign s_axil_bvalid = m_axil_driftadapt_bvalid;
    assign s_axil_bresp = m_axil_driftadapt_bresp;
    assign m_axil_driftadapt_bready = s_axil_bready;
    assign m_axil_driftadapt_arvalid = s_axil_arvalid;
    assign m_axil_driftadapt_araddr = s_axil_araddr;
    assign s_axil_arready = m_axil_driftadapt_arready;
    assign s_axil_rvalid = m_axil_driftadapt_rvalid;
    assign s_axil_rdata = m_axil_driftadapt_rdata;
    assign s_axil_rresp = m_axil_driftadapt_rresp;
    assign m_axil_driftadapt_rready = s_axil_rready;
endmodule
