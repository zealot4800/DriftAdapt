`timescale 1ns/1ps
`include "driftadapt_config.vh"
// Banked counters and registered TV arithmetic. ready rises only when the
// completed window's drift flag is final.
module driftadapt_hist_monitor #(
 parameter WINDOW_SIZE=`DA_WINDOW_SIZE, REFERENCE_SAMPLES=`DA_REFERENCE_SAMPLES,
 parameter TV_NUM=`DA_TV_NUM, TV_DEN=`DA_TV_DEN, CONSECUTIVE=`DA_CONSECUTIVE,
 parameter REFERENCE_MEM_FILE="driftadapt_reference.mem"
)(input wire clk,rst_n, input wire [127:0] bin_ids, input wire accept,
 output wire ready, output reg drift, output reg [31:0] window_id);
 localparam COUNT_BITS=$clog2(WINDOW_SIZE+1)>0 ? $clog2(WINDOW_SIZE+1) : 1;
 localparam REF_BITS=$clog2(REFERENCE_SAMPLES+1)>0 ? $clog2(REFERENCE_SAMPLES+1) : 1;
 localparam CROSS_BITS=$clog2(64'd1*WINDOW_SIZE*REFERENCE_SAMPLES+1)>0 ?
                         $clog2(64'd1*WINDOW_SIZE*REFERENCE_SAMPLES+1) : 1;
 localparam SUM_BITS=$clog2(64'd2*WINDOW_SIZE*REFERENCE_SAMPLES+1)>0 ?
                       $clog2(64'd2*WINDOW_SIZE*REFERENCE_SAMPLES+1) : 1;
 localparam STREAK_BITS=$clog2(CONSECUTIVE+1)>0 ? $clog2(CONSECUTIVE+1) : 1;
 // Integer sum*den > limit is exactly sum > floor(limit/den).
 localparam [SUM_BITS-1:0] TV_LIMIT=(64'd2*TV_NUM*REFERENCE_SAMPLES*WINDOW_SIZE)/TV_DEN;
 localparam INIT=0, COUNT=1, READ_LOCAL=2, SELECT=3, SCALE=4,
            DIFFERENCE=5, ACCUMULATE=6, CHECK=7, FINISH=8;
 reg [3:0] state;
 reg [3:0] scan_bin,scan_feature;
 reg [COUNT_BITS-1:0] samples,count_value;
 reg [STREAK_BITS-1:0] streak;
 reg shifted;
 reg [31:0] reference[0:255];
 reg [REF_BITS-1:0] reference_read,reference_value;
 wire [COUNT_BITS-1:0] bank_read[0:15];
 reg [CROSS_BITS-1:0] observed_scaled,reference_scaled,difference;
 reg [SUM_BITS-1:0] sum;
 initial begin
   if(WINDOW_SIZE<1 || REFERENCE_SAMPLES<1 || TV_DEN<1 ||
      TV_NUM<0 || TV_NUM>TV_DEN || CONSECUTIVE<1)
     $fatal(1,"Invalid histogram monitor parameters");
   $readmemh(REFERENCE_MEM_FILE,reference);
 end
 assign ready=state==COUNT;
 genvar f;
 generate for(f=0;f<16;f=f+1) begin: feature_bank
   (* ram_style="distributed" *) reg [COUNT_BITS-1:0] counts[0:15];
   reg [COUNT_BITS-1:0] selected_count;
   assign bank_read[f]=selected_count;
   always @(posedge clk) begin
     if(rst_n) begin
       if(state==INIT || (state==READ_LOCAL && scan_feature==f))
         counts[scan_bin]<=0;
       else if(accept && ready)
         counts[bin_ids[f*8+2 +: 4]]<=counts[bin_ids[f*8+2 +: 4]]+1'b1;
       if(state==READ_LOCAL) selected_count<=counts[scan_bin];
     end
   end
 end endgenerate
 always @(posedge clk) begin
   if(!rst_n) begin
     state<=INIT;scan_bin<=0;scan_feature<=0;samples<=0;streak<=0;
     shifted<=0;sum<=0;drift<=0;window_id<=0;count_value<=0;
     reference_read<=0;reference_value<=0;observed_scaled<=0;reference_scaled<=0;difference<=0;
   end else case(state)
     INIT: if(scan_bin==15) begin scan_bin<=0;state<=COUNT;end else scan_bin<=scan_bin+1'b1;
     COUNT: if(accept) begin
       if(samples==WINDOW_SIZE-1) begin
         samples<=0;scan_bin<=0;scan_feature<=0;sum<=0;shifted<=0;state<=READ_LOCAL;
       end else samples<=samples+1'b1;
     end
     READ_LOCAL: begin
       reference_read<=reference[{scan_feature,scan_bin}];state<=SELECT;
     end
     SELECT: begin
       count_value<=bank_read[scan_feature];reference_value<=reference_read;state<=SCALE;
     end
     SCALE: begin
       observed_scaled<=count_value*REFERENCE_SAMPLES;
       reference_scaled<=reference_value*WINDOW_SIZE;state<=DIFFERENCE;
     end
     DIFFERENCE: begin
       difference<=observed_scaled>reference_scaled ?
                   observed_scaled-reference_scaled : reference_scaled-observed_scaled;
       state<=ACCUMULATE;
     end
     ACCUMULATE: begin
       sum<=sum+difference;
       if(scan_bin==15) state<=CHECK;
       else begin scan_bin<=scan_bin+1'b1;state<=READ_LOCAL;end
     end
     CHECK: begin
       if(sum>TV_LIMIT) shifted<=1;
       sum<=0;scan_bin<=0;
       if(scan_feature==15) state<=FINISH;
       else begin scan_feature<=scan_feature+1'b1;state<=READ_LOCAL;end
     end
     FINISH: begin
       if(shifted) begin
         if(streak<CONSECUTIVE) streak<=streak+1'b1;
         drift<=streak>=CONSECUTIVE-1;
       end else begin streak<=0;drift<=0;end
       window_id<=window_id+1'b1;state<=COUNT;
     end
     default: state<=INIT;
   endcase
 end
endmodule
