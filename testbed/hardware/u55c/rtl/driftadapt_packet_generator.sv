`timescale 1ns/1ps
`include "driftadapt_config.vh"
// Replays source-preprocessed flow bin IDs. No labels or pseudo labels in ROM.
module driftadapt_packet_generator #(
 parameter SAMPLE_COUNT=`DA_SAMPLE_COUNT, STARTUP_CYCLES=4096, GAP_CYCLES=1,
 parameter SAMPLE_MEM_FILE="driftadapt_samples.mem"
)(input wire clk,rst_n, output wire [511:0] m_axis_tdata,
 output wire m_axis_tvalid, input wire m_axis_tready,
 output wire finished_axis, output reg [63:0] sent_packets_axis);
 reg [127:0] rom[0:SAMPLE_COUNT-1], sample;
 integer index, startup, gap, state;
 initial $readmemh(SAMPLE_MEM_FILE,rom);
 assign m_axis_tdata={272'd0,sample,112'd0};
 assign m_axis_tvalid=state==2;
 assign finished_axis=state==4;
 always @(posedge clk) begin
   if(!rst_n) begin index<=0; startup<=0; gap<=0; state<=0; sample<=0; sent_packets_axis<=0; end
   else case(state)
     0: if(startup>=STARTUP_CYCLES-1) state<=1; else startup<=startup+1;
     1: begin sample<=rom[index]; state<=2; end
     2: if(m_axis_tready) begin
       sent_packets_axis<=sent_packets_axis+1;
       if(index==SAMPLE_COUNT-1) state<=4;
       else begin index<=index+1; gap<=0; state<=3; end
     end
     3: if(gap>=GAP_CYCLES-1) state<=1; else gap<=gap+1;
     default: state<=4;
   endcase
 end
endmodule
