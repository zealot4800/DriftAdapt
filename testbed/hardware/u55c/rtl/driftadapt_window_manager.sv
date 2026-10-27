`timescale 1ns/1ps
`include "driftadapt_config.vh"
// Stable dual windows; private AXI-Lite control. No labels enter the fabric.
module driftadapt_window_manager #(parameter WINDOW_SIZE=`DA_WINDOW_SIZE, SAMPLE_COUNT=`DA_SAMPLE_COUNT)(
 input wire clk,rst_n, input wire [511:0] s_axis_tdata,
 input wire s_axis_tvalid, output wire s_axis_tready,
 input wire drift, loaded, updating, input wire [8:0] staged_words,
 input wire [31:0] model_version, input wire [2:0] model_errors,
 output reg update_begin, stage_valid, commit,
 output reg [8:0] stage_index, output reg [31:0] stage_data,
 output wire [31:0] status,
 input wire s_axil_awvalid, input wire [31:0] s_axil_awaddr, output wire s_axil_awready,
 input wire s_axil_wvalid, input wire [31:0] s_axil_wdata, input wire [3:0] s_axil_wstrb,
 output wire s_axil_wready, output reg s_axil_bvalid, output reg [1:0] s_axil_bresp,
 input wire s_axil_bready, input wire s_axil_arvalid, input wire [31:0] s_axil_araddr,
 output wire s_axil_arready, output reg s_axil_rvalid, output reg [31:0] s_axil_rdata,
 output reg [1:0] s_axil_rresp, input wire s_axil_rready
);
 reg [255:0] records0[0:WINDOW_SIZE-1], records1[0:WINDOW_SIZE-1];
 reg [1:0] full, drift_flags;
 reg bank, release_error;
 integer offset;
 reg [31:0] captured, windows, ids[0:1], first_sample[0:1], expected_release;
 reg [63:0] cycles;
 reg aw_pending,w_pending;
 reg [31:0] awaddr,wdata;
 reg [3:0] wstrb;
 integer read_bank, read_record, read_word;
 reg [255:0] read_value;
 reg [31:0] read_data;
 reg read_ok;
 wire [255:0] record_value={32'd0,s_axis_tdata[53*8 +: 32],24'd0,s_axis_tdata[47*8 +: 8],
                            s_axis_tdata[49*8 +: 32],s_axis_tdata[14*8 +: 128]};
 assign s_axis_tready=!full[bank];
 assign status={24'd0,release_error,|model_errors,updating,loaded,captured==SAMPLE_COUNT,1'b0,full};
 assign s_axil_awready=!aw_pending && !s_axil_bvalid;
 assign s_axil_wready=!w_pending && !s_axil_bvalid;
 assign s_axil_arready=!s_axil_rvalid;
 always @* begin
   read_data=0; read_ok=1; read_bank=0; read_record=0; read_word=0; read_value=0;
   case(s_axil_araddr)
     'h000: read_data=32'h44524654;
     'h004: read_data=32'h00050000;
     'h008: read_data=1;
     'h00c: read_data=status;
     'h014: read_data=ids[0];
     'h018: read_data=ids[1];
     'h01c: read_data=full[0] ? WINDOW_SIZE : 0;
     'h020: read_data=full[1] ? WINDOW_SIZE : 0;
     'h024: read_data=first_sample[0];
     'h028: read_data=first_sample[1];
     'h034: read_data=model_version;
     'h038: read_data=385;
     'h03c: read_data=captured;
     'h040: read_data=0;
     'h044: read_data=cycles[31:0];
     'h048: read_data=cycles[63:32];
     'h060: read_data={30'd0,drift_flags};
     'h064: read_data={28'd0,release_error,model_errors};
     'h068: read_data={23'd0,staged_words};
     'h070: read_data=WINDOW_SIZE;
     'h074: read_data=SAMPLE_COUNT;
     'h07c: read_data=`DA_BUILD_ID;
     default: begin
       if((s_axil_araddr>='h10000 && s_axil_araddr<'h10000+32*WINDOW_SIZE)
         || (s_axil_araddr>='h20000 && s_axil_araddr<'h20000+32*WINDOW_SIZE)) begin
         read_bank=s_axil_araddr[17];
         read_record=(s_axil_araddr & 32'hffff)>>5;
         read_word=(s_axil_araddr>>2)&7;
         if(full[read_bank]) begin
           read_value=read_bank ? records1[read_record] : records0[read_record];
           read_data=read_value[read_word*32 +: 32];
         end else read_ok=0;
       end else read_ok=0;
     end
   endcase
   if(s_axil_araddr[1:0]!=0) read_ok=0;
 end
 always @(posedge clk) begin
   if(!rst_n) begin
     full<=0; drift_flags<=0; bank<=0; offset<=0; captured<=0; windows<=0; cycles<=0;
     ids[0]<=0; ids[1]<=0; first_sample[0]<=0; first_sample[1]<=0; expected_release<=0;
     release_error<=0; aw_pending<=0; w_pending<=0; awaddr<=0; wdata<=0; wstrb<=0;
     s_axil_bvalid<=0; s_axil_bresp<=0; s_axil_rvalid<=0; s_axil_rdata<=0; s_axil_rresp<=0;
     update_begin<=0; stage_valid<=0; commit<=0; stage_index<=0; stage_data<=0;
   end else begin
     if(captured<SAMPLE_COUNT) cycles<=cycles+1;
     update_begin<=0; stage_valid<=0; commit<=0;
     if(s_axis_tvalid && s_axis_tready) begin
       if(bank) records1[offset]<=record_value; else records0[offset]<=record_value;
       captured<=captured+1;
       if(offset==0) first_sample[bank]<=captured;
       if(offset==WINDOW_SIZE-1) begin
         full[bank]<=1; drift_flags[bank]<=drift; ids[bank]<=windows+1;
         windows<=windows+1; offset<=0; bank<=!bank;
       end else offset<=offset+1;
     end
     if(s_axil_awvalid && s_axil_awready) begin awaddr<=s_axil_awaddr; aw_pending<=1; end
     if(s_axil_wvalid && s_axil_wready) begin wdata<=s_axil_wdata; wstrb<=s_axil_wstrb; w_pending<=1; end
     if(s_axil_bvalid && s_axil_bready) s_axil_bvalid<=0;
     if(aw_pending && w_pending && !s_axil_bvalid) begin
       aw_pending<=0; w_pending<=0; s_axil_bvalid<=1; s_axil_bresp<=0;
       if(wstrb!=4'hf || awaddr[1:0]!=0) s_axil_bresp<=2;
       else if(awaddr=='h010) begin
         if(wdata==1 && loaded && !updating) update_begin<=1;
         else if(wdata==2 && updating && staged_words==385) commit<=1;
         else s_axil_bresp<=2;
       end else if(awaddr>='h4000 && awaddr<'h4000+385*4 && updating) begin
         stage_valid<=1; stage_index<=(awaddr-'h4000)>>2; stage_data<=wdata;
       end else if(awaddr=='h080) expected_release<=wdata;
       else if(awaddr=='h084) begin
         if(wdata==1 && full[0] && ids[0]==expected_release) full[0]<=0;
         else if(wdata==2 && full[1] && ids[1]==expected_release) full[1]<=0;
         else begin s_axil_bresp<=2; release_error<=1; end
       end else s_axil_bresp<=2;
     end
     if(s_axil_rvalid && s_axil_rready) s_axil_rvalid<=0;
     if(s_axil_arvalid && s_axil_arready) begin
       s_axil_rvalid<=1; s_axil_rdata<=read_data; s_axil_rresp<=read_ok ? 0 : 2;
     end
   end
 end
endmodule
