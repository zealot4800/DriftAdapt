`timescale 1ns/1ps
`include "driftadapt_config.vh"
// Frozen source reference. TV is evaluated with integer cross multiplication.
module driftadapt_hist_monitor #(
 parameter WINDOW_SIZE=`DA_WINDOW_SIZE, REFERENCE_SAMPLES=`DA_REFERENCE_SAMPLES,
 parameter TV_NUM=`DA_TV_NUM, TV_DEN=`DA_TV_DEN, CONSECUTIVE=`DA_CONSECUTIVE,
 parameter REFERENCE_MEM_FILE="driftadapt_reference.mem"
)(input wire clk,rst_n, input wire [127:0] bin_ids, input wire accept,
 output wire ready, output reg drift, output reg [31:0] window_id);
 reg [31:0] reference[0:255], counts[0:255];
 integer state, index, samples, streak, feature_index;
 reg shifted;
 reg [63:0] sum;
 wire [63:0] a=counts[index]*64'd1*REFERENCE_SAMPLES;
 wire [63:0] b=reference[index]*64'd1*WINDOW_SIZE;
 wire [63:0] difference=a>b ? a-b : b-a;
 wire [63:0] total=sum+difference;
 initial $readmemh(REFERENCE_MEM_FILE,reference);
 assign ready=state==1;
 always @(posedge clk) begin
   if(!rst_n) begin state<=0; index<=0; samples<=0; streak<=0; shifted<=0;
     sum<=0; drift<=0; window_id<=0;
   end else case(state)
     0: begin counts[index]<=0;
       if(index==255) begin index<=0; state<=1; end else index<=index+1;
     end
     1: if(accept) begin
       counts[0+bin_ids[2 +: 4]] <= counts[0+bin_ids[2 +: 4]]+1;
       counts[16+bin_ids[10 +: 4]] <= counts[16+bin_ids[10 +: 4]]+1;
       counts[32+bin_ids[18 +: 4]] <= counts[32+bin_ids[18 +: 4]]+1;
       counts[48+bin_ids[26 +: 4]] <= counts[48+bin_ids[26 +: 4]]+1;
       counts[64+bin_ids[34 +: 4]] <= counts[64+bin_ids[34 +: 4]]+1;
       counts[80+bin_ids[42 +: 4]] <= counts[80+bin_ids[42 +: 4]]+1;
       counts[96+bin_ids[50 +: 4]] <= counts[96+bin_ids[50 +: 4]]+1;
       counts[112+bin_ids[58 +: 4]] <= counts[112+bin_ids[58 +: 4]]+1;
       counts[128+bin_ids[66 +: 4]] <= counts[128+bin_ids[66 +: 4]]+1;
       counts[144+bin_ids[74 +: 4]] <= counts[144+bin_ids[74 +: 4]]+1;
       counts[160+bin_ids[82 +: 4]] <= counts[160+bin_ids[82 +: 4]]+1;
       counts[176+bin_ids[90 +: 4]] <= counts[176+bin_ids[90 +: 4]]+1;
       counts[192+bin_ids[98 +: 4]] <= counts[192+bin_ids[98 +: 4]]+1;
       counts[208+bin_ids[106 +: 4]] <= counts[208+bin_ids[106 +: 4]]+1;
       counts[224+bin_ids[114 +: 4]] <= counts[224+bin_ids[114 +: 4]]+1;
       counts[240+bin_ids[122 +: 4]] <= counts[240+bin_ids[122 +: 4]]+1;
       if(samples==WINDOW_SIZE-1) begin
         samples<=0; index<=0; sum<=0; shifted<=0; state<=2;
       end else samples<=samples+1;
     end
     2: begin
       counts[index]<=0;
       if(index%16==15) begin
         if(total*TV_DEN > 64'd2*TV_NUM*REFERENCE_SAMPLES*WINDOW_SIZE) shifted<=1;
         sum<=0;
       end else sum<=total;
       if(index==255) state<=3; else index<=index+1;
     end
     3: begin
       if(shifted) begin streak<=streak<CONSECUTIVE ? streak+1 : streak;
         drift<=streak+1>=CONSECUTIVE;
       end else begin streak<=0; drift<=0; end
       window_id<=window_id+1; index<=0; state<=1;
     end
   endcase
 end
endmodule
