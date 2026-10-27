`timescale 1ns/1ps
// Public shell aperture is an immutable ABI discovery register. Runtime control
// is exclusively on the axis-clock private JTAG bus, avoiding torn CDC counters.
module driftadapt_control_regs(
 input wire clk,rst_n,
 input wire s_axil_awvalid, input wire [31:0] s_axil_awaddr, output wire s_axil_awready,
 input wire s_axil_wvalid, input wire [31:0] s_axil_wdata, output wire s_axil_wready,
 output reg s_axil_bvalid, output wire [1:0] s_axil_bresp, input wire s_axil_bready,
 input wire s_axil_arvalid, input wire [31:0] s_axil_araddr, output wire s_axil_arready,
 output reg s_axil_rvalid, output reg [31:0] s_axil_rdata,
 output reg [1:0] s_axil_rresp, input wire s_axil_rready
);
 reg aw,w;
 assign s_axil_awready=!aw && !s_axil_bvalid;
 assign s_axil_wready=!w && !s_axil_bvalid;
 assign s_axil_bresp=2;
 assign s_axil_arready=!s_axil_rvalid;
 always @(posedge clk) begin
   if(!rst_n) begin aw<=0;w<=0;s_axil_bvalid<=0;s_axil_rvalid<=0;s_axil_rdata<=0;s_axil_rresp<=0;end
   else begin
     if(s_axil_awvalid && s_axil_awready) aw<=1;
     if(s_axil_wvalid && s_axil_wready) w<=1;
     if(s_axil_bvalid && s_axil_bready) s_axil_bvalid<=0;
     if(aw && w && !s_axil_bvalid) begin aw<=0;w<=0;s_axil_bvalid<=1;end
     if(s_axil_rvalid && s_axil_rready) s_axil_rvalid<=0;
     if(s_axil_arvalid && s_axil_arready) begin
       s_axil_rvalid<=1; s_axil_rresp<=0;
       case(s_axil_araddr[15:0])
         0:s_axil_rdata<=32'h44524654;
         4:s_axil_rdata<=32'h00050000;
         8:s_axil_rdata<=1;
         default:begin s_axil_rdata<=0;s_axil_rresp<=2;end
       endcase
     end
   end
 end
endmodule
