`timescale 1ns/1ps

// DRIFTADAPT online-learning testbed:
//   sample source -> fixed-point DNN -> dual labeling-window buffer.
//
// Traffic and inference stay entirely in fabric. A local control-plane agent
// reads completed windows and returns generated labels through the private
// JTAG AXI master. QDMA/CMAC remain disconnected from the packet datapath.
module driftadapt_u55c_250mhz #(
    parameter integer NUM_QDMA = 1,
    parameter integer NUM_INTF = 2,
    parameter integer NUM_CMAC_PORT = 2,
    parameter integer SAMPLE_COUNT = 16000,
    parameter integer WINDOW_SIZE = 100,
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

    wire [511:0] generated_tdata;
    wire generated_tvalid;
    wire generated_tready;
    wire generator_finished;
    wire [63:0] sent_packets;

    driftadapt_packet_generator #(
        .SAMPLE_COUNT(SAMPLE_COUNT),
        .STARTUP_CYCLES(STARTUP_CYCLES),
        .GAP_CYCLES(GAP_CYCLES),
        .SAMPLE_MEM_FILE(SAMPLE_MEM_FILE)
    ) packet_generator (
        .clk(axis_aclk), .rst_n(axis_aresetn),
        .m_axis_tdata(generated_tdata),
        .m_axis_tvalid(generated_tvalid), .m_axis_tready(generated_tready),
        .finished_axis(generator_finished),
        .sent_packets_axis(sent_packets)
    );

    wire weight_stage_valid;
    wire [7:0] weight_stage_index;
    wire [31:0] weight_stage_data;
    wire weight_commit_request;
    wire weight_commit_ack;
    wire weights_loaded;
    wire active_weight_bank;
    wire [31:0] model_version;
    wire [511:0] classified_tdata;
    wire classified_tvalid;
    wire window_ready;
    wire [63:0] classified_packets;

    driftadapt_dnn_axis #(
        .WEIGHT_MEM_FILE(WEIGHT_MEM_FILE)
    ) dnn (
        .clk(axis_aclk), .rst_n(axis_aresetn),
        .weight_stage_valid_axis(weight_stage_valid),
        .weight_stage_index_axis(weight_stage_index),
        .weight_stage_data_axis(weight_stage_data),
        .weight_commit_request_axis(weight_commit_request),
        .weight_commit_ack_axis(weight_commit_ack),
        .weights_loaded_axis(weights_loaded),
        .active_weight_bank_axis(active_weight_bank),
        .model_version_axis(model_version),
        .classified_packets_axis(classified_packets),
        .s_axis_tdata(generated_tdata),
        .s_axis_tvalid(generated_tvalid), .s_axis_tready(generated_tready),
        .m_axis_tdata(classified_tdata),
        .m_axis_tvalid(classified_tvalid), .m_axis_tready(window_ready)
    );

    wire benchmark_active;
    wire benchmark_complete;
    wire [63:0] total_cycles;
    wire [63:0] first_input_cycle;
    wire [63:0] last_output_cycle;
    wire [63:0] elapsed_cycles;
    wire [63:0] input_stall_cycles;
    wire [63:0] output_stall_cycles;
    wire [63:0] classifier_busy_cycles;
    wire [63:0] latency_sum_cycles;
    wire [63:0] latency_min_cycles;
    wire [63:0] latency_max_cycles;

    driftadapt_benchmark_metrics #(
        .SAMPLE_COUNT(SAMPLE_COUNT)
    ) benchmark_metrics (
        .clk(axis_aclk), .rst_n(axis_aresetn),
        .input_valid(generated_tvalid), .input_ready(generated_tready),
        .output_valid(classified_tvalid), .output_ready(window_ready),
        .benchmark_active_axis(benchmark_active),
        .benchmark_complete_axis(benchmark_complete),
        .total_cycles_axis(total_cycles),
        .first_input_cycle_axis(first_input_cycle),
        .last_output_cycle_axis(last_output_cycle),
        .elapsed_cycles_axis(elapsed_cycles),
        .input_stall_cycles_axis(input_stall_cycles),
        .output_stall_cycles_axis(output_stall_cycles),
        .classifier_busy_cycles_axis(classifier_busy_cycles),
        .latency_sum_cycles_axis(latency_sum_cycles),
        .latency_min_cycles_axis(latency_min_cycles),
        .latency_max_cycles_axis(latency_max_cycles)
    );

    // Private JTAG AXI-Lite master used by the local labeling agent. The
    // interface does not enumerate PCIe and is outside the measured stream.
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

    wire online_complete;
    wire [63:0] total_captured;
    wire [63:0] total_labeled;
    wire [63:0] proxy_tp;
    wire [63:0] proxy_tn;
    wire [63:0] proxy_fp;
    wire [63:0] proxy_fn;

    driftadapt_window_manager #(
        .DATASET_COUNT(SAMPLE_COUNT), .WINDOW_SIZE(WINDOW_SIZE)
    ) window_manager (
        .clk(axis_aclk), .rst_n(axis_aresetn),
        .s_axis_tdata(classified_tdata),
        .s_axis_tvalid(classified_tvalid), .s_axis_tready(window_ready),
        .generator_active_axis(benchmark_active),
        .generator_finished_axis(generator_finished),
        .sent_packets_axis(sent_packets),
        .classified_packets_axis(classified_packets),
        .elapsed_cycles_axis(elapsed_cycles),
        .input_stall_cycles_axis(input_stall_cycles),
        .output_stall_cycles_axis(output_stall_cycles),
        .classifier_busy_cycles_axis(classifier_busy_cycles),
        .latency_sum_cycles_axis(latency_sum_cycles),
        .latency_min_cycles_axis(latency_min_cycles),
        .latency_max_cycles_axis(latency_max_cycles),
        .total_cycles_axis(total_cycles),
        .online_complete_axis(online_complete),
        .total_captured_axis(total_captured),
        .total_labeled_axis(total_labeled),
        .proxy_true_positive_axis(proxy_tp),
        .proxy_true_negative_axis(proxy_tn),
        .proxy_false_positive_axis(proxy_fp),
        .proxy_false_negative_axis(proxy_fn),
        .weight_stage_valid_axis(weight_stage_valid),
        .weight_stage_index_axis(weight_stage_index),
        .weight_stage_data_axis(weight_stage_data),
        .weight_commit_request_axis(weight_commit_request),
        .weight_commit_ack_axis(weight_commit_ack),
        .weights_loaded_axis(weights_loaded),
        .active_weight_bank_axis(active_weight_bank),
        .model_version_axis(model_version),
        .s_axil_awvalid(jtag_awvalid), .s_axil_awaddr(jtag_awaddr),
        .s_axil_awready(jtag_awready), .s_axil_wvalid(jtag_wvalid),
        .s_axil_wdata(jtag_wdata), .s_axil_wready(jtag_wready),
        .s_axil_bvalid(jtag_bvalid), .s_axil_bresp(jtag_bresp),
        .s_axil_bready(jtag_bready), .s_axil_arvalid(jtag_arvalid),
        .s_axil_araddr(jtag_araddr), .s_axil_arready(jtag_arready),
        .s_axil_rvalid(jtag_rvalid), .s_axil_rdata(jtag_rdata),
        .s_axil_rresp(jtag_rresp), .s_axil_rready(jtag_rready)
    );

    // A read-only mirror is retained on the OpenNIC user AXI-Lite aperture for
    // diagnostics. Online window and model mutations use the private JTAG path.
    driftadapt_control_regs #(
        .DATASET_COUNT(SAMPLE_COUNT)
    ) shell_status_regs (
        .s_axil_awvalid(s_axil_awvalid), .s_axil_awaddr(s_axil_awaddr),
        .s_axil_awready(s_axil_awready), .s_axil_wvalid(s_axil_wvalid),
        .s_axil_wdata(s_axil_wdata), .s_axil_wready(s_axil_wready),
        .s_axil_bvalid(s_axil_bvalid), .s_axil_bresp(s_axil_bresp),
        .s_axil_bready(s_axil_bready), .s_axil_arvalid(s_axil_arvalid),
        .s_axil_araddr(s_axil_araddr), .s_axil_arready(s_axil_arready),
        .s_axil_rvalid(s_axil_rvalid), .s_axil_rdata(s_axil_rdata),
        .s_axil_rresp(s_axil_rresp), .s_axil_rready(s_axil_rready),
        .axil_aclk(axil_aclk), .axil_aresetn(axil_aresetn),
        .generator_active_axis(benchmark_active),
        .generator_finished_axis(generator_finished),
        .result_complete_axis(online_complete),
        .sent_packets_axis(sent_packets),
        .received_packets_axis(total_captured),
        .valid_packets_axis(total_labeled),
        .true_positive_axis(proxy_tp), .true_negative_axis(proxy_tn),
        .false_positive_axis(proxy_fp), .false_negative_axis(proxy_fn),
        .malformed_packets_axis(64'd0), .ignored_packets_axis(64'd0),
        .classified_packets_axis(classified_packets),
        .elapsed_cycles_axis(elapsed_cycles),
        .input_stall_cycles_axis(input_stall_cycles),
        .output_stall_cycles_axis(output_stall_cycles),
        .classifier_busy_cycles_axis(classifier_busy_cycles),
        .latency_sum_cycles_axis(latency_sum_cycles),
        .latency_min_cycles_axis(latency_min_cycles),
        .latency_max_cycles_axis(latency_max_cycles),
        .first_input_cycle_axis(first_input_cycle),
        .last_output_cycle_axis(last_output_cycle),
        .total_cycles_axis(total_cycles)
    );

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

    wire unused_interfaces = ^{
        benchmark_complete, m_axis_adap_tx_250mhz_tready,
        s_axis_adap_rx_250mhz_tvalid, s_axis_adap_rx_250mhz_tdata,
        s_axis_adap_rx_250mhz_tkeep, s_axis_adap_rx_250mhz_tlast,
        s_axis_adap_rx_250mhz_tuser_size, s_axis_adap_rx_250mhz_tuser_src,
        s_axis_adap_rx_250mhz_tuser_dst, s_axis_qdma_h2c_tvalid,
        s_axis_qdma_h2c_tdata, s_axis_qdma_h2c_tkeep, s_axis_qdma_h2c_tlast,
        s_axis_qdma_h2c_tuser_size, s_axis_qdma_h2c_tuser_src,
        s_axis_qdma_h2c_tuser_dst, m_axis_qdma_c2h_tready,
        jtag_awprot, jtag_wstrb, jtag_arprot
    };
endmodule
