wire axil_driftadapt_awvalid;
wire [31:0] axil_driftadapt_awaddr;
wire axil_driftadapt_awready;
wire axil_driftadapt_wvalid;
wire [31:0] axil_driftadapt_wdata;
wire axil_driftadapt_wready;
wire axil_driftadapt_bvalid;
wire [1:0] axil_driftadapt_bresp;
wire axil_driftadapt_bready;
wire axil_driftadapt_arvalid;
wire [31:0] axil_driftadapt_araddr;
wire axil_driftadapt_arready;
wire axil_driftadapt_rvalid;
wire [31:0] axil_driftadapt_rdata;
wire [1:0] axil_driftadapt_rresp;
wire axil_driftadapt_rready;

box_250mhz_address_map address_map_inst (
    .s_axil_awvalid(s_axil_awvalid), .s_axil_awaddr(s_axil_awaddr),
    .s_axil_awready(s_axil_awready), .s_axil_wvalid(s_axil_wvalid),
    .s_axil_wdata(s_axil_wdata), .s_axil_wready(s_axil_wready),
    .s_axil_bvalid(s_axil_bvalid), .s_axil_bresp(s_axil_bresp),
    .s_axil_bready(s_axil_bready), .s_axil_arvalid(s_axil_arvalid),
    .s_axil_araddr(s_axil_araddr), .s_axil_arready(s_axil_arready),
    .s_axil_rvalid(s_axil_rvalid), .s_axil_rdata(s_axil_rdata),
    .s_axil_rresp(s_axil_rresp), .s_axil_rready(s_axil_rready),
    .m_axil_driftadapt_awvalid(axil_driftadapt_awvalid),
    .m_axil_driftadapt_awaddr(axil_driftadapt_awaddr),
    .m_axil_driftadapt_awready(axil_driftadapt_awready),
    .m_axil_driftadapt_wvalid(axil_driftadapt_wvalid),
    .m_axil_driftadapt_wdata(axil_driftadapt_wdata),
    .m_axil_driftadapt_wready(axil_driftadapt_wready),
    .m_axil_driftadapt_bvalid(axil_driftadapt_bvalid),
    .m_axil_driftadapt_bresp(axil_driftadapt_bresp),
    .m_axil_driftadapt_bready(axil_driftadapt_bready),
    .m_axil_driftadapt_arvalid(axil_driftadapt_arvalid),
    .m_axil_driftadapt_araddr(axil_driftadapt_araddr),
    .m_axil_driftadapt_arready(axil_driftadapt_arready),
    .m_axil_driftadapt_rvalid(axil_driftadapt_rvalid),
    .m_axil_driftadapt_rdata(axil_driftadapt_rdata),
    .m_axil_driftadapt_rresp(axil_driftadapt_rresp),
    .m_axil_driftadapt_rready(axil_driftadapt_rready)
);
