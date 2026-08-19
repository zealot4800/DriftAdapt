initial begin
    if (USE_PHYS_FUNC == 0 || NUM_QDMA != 1 || NUM_PHYS_FUNC != 2 || NUM_CMAC_PORT != 2)
        $fatal(1, "DRIFTADAPT U55C requires physical functions, one QDMA, two PFs, and two CMACs");
end

localparam C_NUM_USER_BLOCK = 1;
assign mod_rst_done[15:C_NUM_USER_BLOCK] = {(16-C_NUM_USER_BLOCK){1'b1}};

driftadapt_u55c_250mhz #(
    .NUM_QDMA(NUM_QDMA), .NUM_INTF(NUM_PHYS_FUNC), .NUM_CMAC_PORT(NUM_CMAC_PORT)
) driftadapt_u55c_inst (
    .s_axil_awvalid(axil_driftadapt_awvalid), .s_axil_awaddr(axil_driftadapt_awaddr),
    .s_axil_awready(axil_driftadapt_awready), .s_axil_wvalid(axil_driftadapt_wvalid),
    .s_axil_wdata(axil_driftadapt_wdata), .s_axil_wready(axil_driftadapt_wready),
    .s_axil_bvalid(axil_driftadapt_bvalid), .s_axil_bresp(axil_driftadapt_bresp),
    .s_axil_bready(axil_driftadapt_bready), .s_axil_arvalid(axil_driftadapt_arvalid),
    .s_axil_araddr(axil_driftadapt_araddr), .s_axil_arready(axil_driftadapt_arready),
    .s_axil_rvalid(axil_driftadapt_rvalid), .s_axil_rdata(axil_driftadapt_rdata),
    .s_axil_rresp(axil_driftadapt_rresp), .s_axil_rready(axil_driftadapt_rready),
    .s_axis_qdma_h2c_tvalid(s_axis_qdma_h2c_tvalid),
    .s_axis_qdma_h2c_tdata(s_axis_qdma_h2c_tdata),
    .s_axis_qdma_h2c_tkeep(s_axis_qdma_h2c_tkeep),
    .s_axis_qdma_h2c_tlast(s_axis_qdma_h2c_tlast),
    .s_axis_qdma_h2c_tuser_size(s_axis_qdma_h2c_tuser_size),
    .s_axis_qdma_h2c_tuser_src(s_axis_qdma_h2c_tuser_src),
    .s_axis_qdma_h2c_tuser_dst(s_axis_qdma_h2c_tuser_dst),
    .s_axis_qdma_h2c_tready(s_axis_qdma_h2c_tready),
    .m_axis_qdma_c2h_tvalid(m_axis_qdma_c2h_tvalid),
    .m_axis_qdma_c2h_tdata(m_axis_qdma_c2h_tdata),
    .m_axis_qdma_c2h_tkeep(m_axis_qdma_c2h_tkeep),
    .m_axis_qdma_c2h_tlast(m_axis_qdma_c2h_tlast),
    .m_axis_qdma_c2h_tuser_size(m_axis_qdma_c2h_tuser_size),
    .m_axis_qdma_c2h_tuser_src(m_axis_qdma_c2h_tuser_src),
    .m_axis_qdma_c2h_tuser_dst(m_axis_qdma_c2h_tuser_dst),
    .m_axis_qdma_c2h_tready(m_axis_qdma_c2h_tready),
    .m_axis_adap_tx_250mhz_tvalid(m_axis_adap_tx_250mhz_tvalid),
    .m_axis_adap_tx_250mhz_tdata(m_axis_adap_tx_250mhz_tdata),
    .m_axis_adap_tx_250mhz_tkeep(m_axis_adap_tx_250mhz_tkeep),
    .m_axis_adap_tx_250mhz_tlast(m_axis_adap_tx_250mhz_tlast),
    .m_axis_adap_tx_250mhz_tuser_size(m_axis_adap_tx_250mhz_tuser_size),
    .m_axis_adap_tx_250mhz_tuser_src(m_axis_adap_tx_250mhz_tuser_src),
    .m_axis_adap_tx_250mhz_tuser_dst(m_axis_adap_tx_250mhz_tuser_dst),
    .m_axis_adap_tx_250mhz_tready(m_axis_adap_tx_250mhz_tready),
    .s_axis_adap_rx_250mhz_tvalid(s_axis_adap_rx_250mhz_tvalid),
    .s_axis_adap_rx_250mhz_tdata(s_axis_adap_rx_250mhz_tdata),
    .s_axis_adap_rx_250mhz_tkeep(s_axis_adap_rx_250mhz_tkeep),
    .s_axis_adap_rx_250mhz_tlast(s_axis_adap_rx_250mhz_tlast),
    .s_axis_adap_rx_250mhz_tuser_size(s_axis_adap_rx_250mhz_tuser_size),
    .s_axis_adap_rx_250mhz_tuser_src(s_axis_adap_rx_250mhz_tuser_src),
    .s_axis_adap_rx_250mhz_tuser_dst(s_axis_adap_rx_250mhz_tuser_dst),
    .s_axis_adap_rx_250mhz_tready(s_axis_adap_rx_250mhz_tready),
    .mod_rstn(mod_rstn[0]), .mod_rst_done(mod_rst_done[0]),
    .axil_aclk(axil_aclk), .axis_aclk(axis_aclk)
);
