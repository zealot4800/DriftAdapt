`timescale 1ns/1ps
// ABI 5: packed evidence LUT + int8 16->16->8->2 MLP, full-image atomic updates.
module driftadapt_dnn_axis #(parameter WEIGHT_MEM_FILE="driftadapt_weights.mem") (
 input wire clk, rst_n,
 input wire update_begin, stage_valid, commit,
 input wire [8:0] stage_index, input wire [31:0] stage_data,
 output reg loaded, updating, output reg [8:0] staged_words,
 output reg [31:0] model_version, output reg [2:0] errors,
 input wire [511:0] s_axis_tdata, input wire s_axis_tvalid, output wire s_axis_tready,
 output wire [511:0] m_axis_tdata, output wire m_axis_tvalid, input wire m_axis_tready
);
 localparam BOOT=0, IDLE=1, ENCODE=2, BEGIN_NEURON=3, OPERANDS=4,
            MAC=5, STORE=6, COPY=7, OUTPUT_PACKET=8,
            QSUM=9, QFULL=10, QROUND=11, ACTIVATE=12;
 reg [3:0] state;
 reg [31:0] rom[0:384];
 reg [31:0] bank0[0:384], bank1[0:384];
 reg active, pending;
 reg [384:0] written;
 reg [8:0] boot_index;
 reg [4:0] feature_index,neuron,copy_index;
 reg [1:0] layer;
 reg signed [8:0] activation[0:15];
 reg [7:0] next_activation[0:15];
 reg signed [8:0] operand_a;
 reg signed [7:0] operand_w;
 reg [31:0] requant_multiplier,p00,p01,p10,p11;
 reg [63:0] qlo,qhi,qproduct,scaled;
 reg signed [63:0] accumulator, first_logit;
 reg [511:0] packet;
 wire [4:0] nin=layer==2 ? 8 : 16;
 wire [4:0] nout=layer==0 ? 16 : (layer==1 ? 8 : 2);
 wire [10:0] weight_offset=layer==0 ? 1024 : (layer==1 ? 1280 : 1408);
 wire [8:0] bias_offset=layer==0 ? 356 : (layer==1 ? 372 : 380);
 wire [10:0] lut_index=feature_index*64+packet[(14+feature_index)*8 +: 8];
 wire [10:0] weight_index=weight_offset+(layer==2 ? neuron*8 : neuron*16)+feature_index;
 wire [31:0] lut_word=active ? bank1[lut_index/4] : bank0[lut_index/4];
 wire [31:0] weight_word=active ? bank1[weight_index/4] : bank0[weight_index/4];
 wire signed [7:0] lut_byte=lut_word[(lut_index%4)*8 +: 8];
 wire signed [7:0] weight_byte=weight_word[(weight_index%4)*8 +: 8];
 wire signed [31:0] bias_word=active ? bank1[bias_offset+neuron] : bank0[bias_offset+neuron];
 wire [31:0] multiplier_word=active ? bank1[382+layer] : bank0[382+layer];
 wire signed [31:0] threshold_word=active ? bank1[384] : bank0[384];
 wire signed [63:0] margin=accumulator-first_logit;
 initial $readmemh(WEIGHT_MEM_FILE, rom);
 assign s_axis_tready=loaded && state==IDLE && !updating && !pending && !update_begin && !commit;
 assign m_axis_tvalid=state==OUTPUT_PACKET;
 assign m_axis_tdata=packet;
 always @(posedge clk) begin
   if(!rst_n) begin
     state<=BOOT; boot_index<=0; loaded<=0; active<=0; pending<=0;
     updating<=0; written<=0; staged_words<=0; model_version<=0; errors<=0;
     packet<=0; layer<=0; feature_index<=0; neuron<=0; copy_index<=0;
     accumulator<=0; first_logit<=0; operand_a<=0; operand_w<=0;
     requant_multiplier<=0;p00<=0;p01<=0;p10<=0;p11<=0;qlo<=0;qhi<=0;qproduct<=0;scaled<=0;
   end else begin
     if(update_begin && loaded && !updating && !pending) begin
       updating<=1; written<=0; staged_words<=0; errors[0]<=0;
     end
     if(stage_valid) begin
       if(updating && !pending && stage_index<385) begin
         if(active) bank0[stage_index]<=stage_data; else bank1[stage_index]<=stage_data;
         if(!written[stage_index]) staged_words<=staged_words+1'b1;
         written[stage_index]<=1;
       end else errors[0]<=1;
     end
     if(commit) begin
       if(updating && &written && !errors[0]) pending<=1;
       else begin errors[0]<=1; updating<=0; end
     end
     case(state)
       BOOT: begin
         bank0[boot_index]<=rom[boot_index]; bank1[boot_index]<=rom[boot_index];
         if(boot_index==384) begin loaded<=1; state<=IDLE; end
         else boot_index<=boot_index+1;
       end
       IDLE: begin
         if(pending) begin
           active<=!active; model_version<=model_version+1; pending<=0; updating<=0;
         end else if(s_axis_tvalid && s_axis_tready) begin
           packet<=s_axis_tdata; packet[53*8 +: 32]<=model_version;
           feature_index<=0; layer<=0; state<=ENCODE;
         end
       end
       ENCODE: begin
         activation[feature_index]<=lut_byte;
         if(packet[(14+feature_index)*8 +: 8]>63) errors[1]<=1;
         if(feature_index==15) begin neuron<=0; state<=BEGIN_NEURON; end
         else feature_index<=feature_index+1;
       end
       BEGIN_NEURON: begin
         accumulator<=bias_word; feature_index<=0; state<=OPERANDS;
         requant_multiplier<=multiplier_word;
       end
       OPERANDS: begin
         operand_a<=activation[feature_index];
         operand_w<=weight_byte; state<=MAC;
       end
       MAC: begin
         accumulator<=accumulator+operand_a*operand_w;
         if(feature_index==nin-1) state<=STORE;
         else begin feature_index<=feature_index+1; state<=OPERANDS; end
       end
       STORE: begin
         if(accumulator < -64'sd2147483648 || accumulator > 64'sd2147483647) errors[2]<=1;
         if(layer<2) begin
           // Four registered 16x16 products avoid a wide unpipelined multiplier.
           p00<=accumulator>0 ? accumulator[15:0]*requant_multiplier[15:0] : 0;
           p01<=accumulator>0 ? accumulator[15:0]*requant_multiplier[31:16] : 0;
           p10<=accumulator>0 ? accumulator[31:16]*requant_multiplier[15:0] : 0;
           p11<=accumulator>0 ? accumulator[31:16]*requant_multiplier[31:16] : 0;
           state<=QSUM;
         end else if(neuron==0) begin
           first_logit<=accumulator; neuron<=1; state<=BEGIN_NEURON;
         end else begin
           if(margin < -64'sd2147483648 || margin > 64'sd2147483647) errors[2]<=1;
           packet[47*8 +: 8]<=margin >= threshold_word ? 8'd1 : 8'd0;
           packet[48*8 +: 8]<=8'd1;
           packet[49*8 +: 32]<=margin[31:0]; state<=OUTPUT_PACKET;
         end
       end
       QSUM: begin
         qlo<={32'd0,p00}+({32'd0,p01}<<16);
         qhi<=({32'd0,p10}<<16)+({32'd0,p11}<<32);state<=QFULL;
       end
       QFULL: begin qproduct<=qlo+qhi;state<=QROUND;end
       QROUND: begin scaled<=(qproduct+64'd8388608)>>24;state<=ACTIVATE;end
       ACTIVATE: begin
         next_activation[neuron]<=scaled>255 ? 8'd255 : scaled[7:0];
         if(neuron==nout-1) begin copy_index<=0;state<=COPY;end
         else begin neuron<=neuron+1;state<=BEGIN_NEURON;end
       end
       COPY: begin
         activation[copy_index]<=$signed({1'b0,next_activation[copy_index]});
         if(copy_index==nout-1) begin layer<=layer+1; neuron<=0; state<=BEGIN_NEURON; end
         else copy_index<=copy_index+1;
       end
       OUTPUT_PACKET: if(m_axis_tready) state<=IDLE;
       default: state<=BOOT;
     endcase
   end
 end
endmodule
