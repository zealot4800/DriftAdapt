#!/usr/bin/env python3
"""Run XSim on actual encoded-MLP RTL, complete replay and atomic-update checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from driftadapt_runtime import ASSETS, fixed_inference, read_words, write_words
import numpy as np

BENCH = r"""
`timescale 1ns/1ps
`include "driftadapt_config.vh"
module tb;
 reg clk=0; always #2 clk=~clk;
 reg rst=0;
 wire [511:0] packet,output_packet;
 wire valid,ready,core_ready,out_valid,out_ready,hist_ready,drift,finished,loaded,updating;
 wire [63:0] sent;
 wire [31:0] model_version,hist_window,status;
 wire [8:0] staged;
 wire [2:0] errors;
 reg begin_update=0,stage_valid=0,commit=0;
 reg [8:0] stage_index=0;reg [31:0] stage_data=0;
 reg awvalid=0,wvalid=0,bready=1,arvalid=0,rready=1;
 reg [31:0] awaddr=0,wdata=0,araddr=0;
 wire awready,wready,bvalid,arready,rvalid;
 wire [1:0] bresp,rresp;
 wire [31:0] rdata;
 wire manager_begin,manager_stage,manager_commit;
 wire [8:0] manager_index;wire [31:0] manager_data;
 integer captured=0,checks=0,version_checks=0,wait_count=0;
 reg checked_update=0;
 reg [31:0] golden_margin[0:`DA_SAMPLE_COUNT-1],golden_prediction[0:`DA_SAMPLE_COUNT-1];
 reg [31:0] golden_drift[0:`DA_SAMPLE_COUNT/`DA_WINDOW_SIZE-1],model[0:384];
 reg [127:0] golden_bins[0:`DA_SAMPLE_COUNT-1];
 assign ready=core_ready && hist_ready;
 driftadapt_packet_generator source(.clk(clk),.rst_n(rst),.m_axis_tdata(packet),
 .m_axis_tvalid(valid),.m_axis_tready(ready),.finished_axis(finished),.sent_packets_axis(sent));
 driftadapt_hist_monitor hist(.clk(clk),.rst_n(rst),.bin_ids(packet[112 +: 128]),
 .accept(valid && ready),.ready(hist_ready),.drift(drift),.window_id(hist_window));
 driftadapt_dnn_axis core(.clk(clk),.rst_n(rst),.update_begin(begin_update || manager_begin),
 .stage_valid(stage_valid || manager_stage),.stage_index(stage_valid ? stage_index : manager_index),
 .stage_data(stage_valid ? stage_data : manager_data),.commit(commit || manager_commit),
 .loaded(loaded),.updating(updating),.staged_words(staged),.model_version(model_version),.errors(errors),
 .s_axis_tdata(packet),.s_axis_tvalid(valid && hist_ready),.s_axis_tready(core_ready),
 .m_axis_tdata(output_packet),.m_axis_tvalid(out_valid),.m_axis_tready(out_ready));
 driftadapt_window_manager manager(.clk(clk),.rst_n(rst),
 .s_axis_tdata(output_packet),.s_axis_tvalid(out_valid),.s_axis_tready(out_ready),
 .drift(drift),.loaded(loaded),.updating(updating),.staged_words(staged),.model_version(model_version),
 .model_errors(errors),.update_begin(manager_begin),.stage_valid(manager_stage),
 .commit(manager_commit),.stage_index(manager_index),.stage_data(manager_data),.status(status),
 .s_axil_awvalid(awvalid),.s_axil_awaddr(awaddr),.s_axil_awready(awready),
 .s_axil_wvalid(wvalid),.s_axil_wdata(wdata),.s_axil_wstrb(4'hf),.s_axil_wready(wready),
 .s_axil_bvalid(bvalid),.s_axil_bresp(bresp),.s_axil_bready(bready),
 .s_axil_arvalid(arvalid),.s_axil_araddr(araddr),.s_axil_arready(arready),
 .s_axil_rvalid(rvalid),.s_axil_rdata(rdata),.s_axil_rresp(rresp),.s_axil_rready(rready));
 task read_word(input [31:0] addr,output [31:0] value);
 begin
   @(negedge clk);araddr=addr;arvalid=1;
   do @(posedge clk); while(!arready);
   @(negedge clk);arvalid=0;
   wait(rvalid);value=rdata;
   if(rresp!=0) $fatal(1,"AXI read error %h",addr);
   @(posedge clk);@(negedge clk);
 end endtask
 task write_word(input [31:0] addr,input [31:0] value,input [1:0] expected);
 begin
   // Deliberately separate AW and W handshakes.
   @(negedge clk);awaddr=addr;awvalid=1;
   do @(posedge clk);while(!awready);
   @(negedge clk);awvalid=0;
   repeat(2) @(negedge clk);
   wdata=value;wvalid=1;
   do @(posedge clk);while(!wready);
   @(negedge clk);wvalid=0;
   wait(bvalid);if(bresp!=expected) $fatal(1,"AXI write response %h",addr);
   @(posedge clk);@(negedge clk);
 end endtask
 task load_image(input bit change_threshold);
 integer j;
 begin
   @(negedge clk);begin_update=1;
   @(negedge clk);begin_update=0;
   for(j=0;j<385;j=j+1) begin
     stage_valid=1;stage_index=j;stage_data=(change_threshold && j==384) ? 32'h80000000 : model[j];
     @(negedge clk);
   end
   stage_valid=0;commit=1;
   @(negedge clk);commit=0;
   wait(!updating);
 end endtask
 // Check every handshake, including the packet that was in flight at commit.
 always @(posedge clk) if(rst && out_valid && out_ready) begin
   if(output_packet[49*8 +: 32] !== golden_margin[captured]) $fatal(1,"Margin mismatch %d",captured);
   if(output_packet[14*8 +: 128] !== golden_bins[captured]) $fatal(1,"Bin order mismatch");
   if(output_packet[53*8 +: 32]==1) begin
     if(output_packet[47*8 +: 8]!=1) $fatal(1,"Updated threshold not applied");
     version_checks=version_checks+1;
   end else if(output_packet[47*8 +: 8] !== golden_prediction[captured][7:0]) $fatal(1,"Prediction mismatch");
   captured=captured+1;
 end
 initial begin
   $readmemh("margins.mem",golden_margin);$readmemh("predictions.mem",golden_prediction);
   $readmemh("drift.mem",golden_drift);$readmemh("driftadapt_samples.mem",golden_bins);
   $readmemh("driftadapt_weights.mem",model);
   repeat(5) @(negedge clk);rst=1;
   wait(loaded);
   // Incomplete images are rejected, without changing the active model.
   @(negedge clk);begin_update=1;@(negedge clk);begin_update=0;
   stage_valid=1;stage_index=0;stage_data=model[0];@(negedge clk);stage_valid=0;
   commit=1;@(negedge clk);commit=0;
   repeat(3) @(negedge clk);
   if(model_version!=0 || errors[0]!=1 || updating) $fatal(1,"Incomplete update accepted");
   // Corrupt shadow image, then reset: BOTH banks must return to immutable ROM.
   load_image(1);
   @(negedge clk);rst=0;repeat(5) @(negedge clk);rst=1;
   wait(loaded);if(core.bank0[384]!==model[384] || core.bank1[384]!==model[384]) $fatal(1,"Reset did not reload ROM");
   // Wait for an in-flight inference, then commit a complete new threshold.
   wait(valid && ready);wait(!core_ready);load_image(1);
   wait(model_version==1);
 end
 integer bank,id,i,j,base,first;
 reg [31:0] value,again;
 initial begin
   wait(rst);wait(loaded);wait(!rst);wait(rst);wait(loaded);
   while(checks<`DA_SAMPLE_COUNT) begin
     wait(status[1:0]!=0);
     // Leave both banks full for multiple clocks: stable data and backpressure.
     if(checks==0) begin
       wait(status[1:0]==3);repeat(100) @(posedge clk);
       if(out_ready) $fatal(1,"Full windows did not backpressure");
     end
     if(status[1:0]==3) begin
       read_word('h14,value);read_word('h18,again);bank=value<again ? 0 : 1;
     end else bank=status[0] ? 0 : 1;
     read_word('h14+bank*4,value);id=value;
     read_word('h24+bank*4,value);first=value;
     if(id!=checks/`DA_WINDOW_SIZE+1 || first!=checks) $fatal(1,"Window order");
     read_word('h60,value);
     if(((value>>bank)&1)!==golden_drift[id-1]) $fatal(1,"Histogram drift mismatch %d",id);
     base=(bank+1)*'h10000;
     for(i=0;i<`DA_WINDOW_SIZE;i=i+1) begin
       for(j=0;j<4;j=j+1) begin
         read_word(base+i*32+j*4,value);
         if(value!==golden_bins[checks+i][j*32 +: 32]) $fatal(1,"Window source data mismatch");
       end
       read_word(base+i*32+16,value);
       if(value!==golden_margin[checks+i]) $fatal(1,"Window margin mismatch");
     end
     if(checks==0) begin
       // Wrong window release must fail, leaving immutable data intact.
       write_word('h80,id+1000,0);write_word('h84,1<<bank,2);
       read_word(base,value);repeat(20) @(posedge clk);read_word(base,again);
       if(value!==again || !status[bank]) $fatal(1,"Invalid release changed window");
     end
     write_word('h80,id,0);write_word('h84,1<<bank,0);
     checks=checks+`DA_WINDOW_SIZE;
     if(checks==2*`DA_WINDOW_SIZE) begin
       load_image(0);wait(model_version==2);
     end
   end
   if(captured!=`DA_SAMPLE_COUNT || version_checks==0 || errors!=0) $fatal(1,"Incomplete validation");
   $display("RTL_PASS samples=%0d updated_predictions=%0d windows=%0d",captured,version_checks,checks/`DA_WINDOW_SIZE);
   $finish;
 end
 initial begin #200000000; $fatal(1,"Simulation timeout");end
endmodule
"""

# Interface-only substitutes for shell-generated IP; not exercised by tb.
SHELL_STUBS = r"""
module generic_reset #(parameter NUM_INPUT_CLK=2,RESET_DURATION=100)(
 input mod_rstn,output mod_rst_done,input [NUM_INPUT_CLK-1:0] clk,
 output [NUM_INPUT_CLK-1:0] rstn);
 assign rstn={NUM_INPUT_CLK{mod_rstn}};assign mod_rst_done=mod_rstn;
endmodule
module driftadapt_jtag_axi(
 input aclk,aresetn,
 output [31:0] m_axi_awaddr,output [2:0] m_axi_awprot,output m_axi_awvalid,input m_axi_awready,
 output [31:0] m_axi_wdata,output [3:0] m_axi_wstrb,output m_axi_wvalid,input m_axi_wready,
 input [1:0] m_axi_bresp,input m_axi_bvalid,output m_axi_bready,
 output [31:0] m_axi_araddr,output [2:0] m_axi_arprot,output m_axi_arvalid,input m_axi_arready,
 input [31:0] m_axi_rdata,input [1:0] m_axi_rresp,input m_axi_rvalid,output m_axi_rready);
endmodule
"""


def verify(args):
    assets = args.assets.resolve()
    build = args.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((assets/'manifest.json').read_text())
    words = read_words(assets/'driftadapt_weights.mem')
    bins = np.asarray([list(bytes.fromhex(line)[::-1]) for line in
                       (assets/'driftadapt_samples.mem').read_text().splitlines()])
    pred, margins = fixed_inference(bins,words)
    write_words(build/'margins.mem',margins)
    write_words(build/'predictions.mem',pred)
    reference = read_words(assets/'driftadapt_reference.mem').reshape(16,16).astype(np.int64)
    nref = int(reference[0].sum())
    size = manifest['window_size']
    streak=0
    flags=[]
    for start in range(0,len(bins),size):
        current=np.stack([np.bincount(bins[start:start+size,f]//4,minlength=16) for f in range(16)])
        diff=np.abs(current*nref-reference*size).sum(axis=1)
        shifted=bool(np.any(diff>2*manifest['drift']['threshold']*size*nref))
        streak=streak+1 if shifted else 0
        flags.append(int(streak>=manifest['drift']['consecutive_windows']))
    write_words(build/'drift.mem',flags)
    for name in manifest['files']:
        (build/name).write_bytes((assets/name).read_bytes())
    (build/'tb.sv').write_text(BENCH)
    (build/'shell_stubs.sv').write_text(SHELL_STUBS)
    rtl=assets.parent/'rtl'
    commands=[
      [str(args.vivado_bin/'xvlog'),'--sv','-i',str(build),*[str(rtl/name) for name in
        ('driftadapt_dnn_axis.sv','driftadapt_hist_monitor.sv','driftadapt_packet_generator.sv',
         'driftadapt_window_manager.sv','driftadapt_control_regs.sv')],
       str(assets.parent/'plugin/driftadapt_u55c_250mhz.sv'),'shell_stubs.sv','tb.sv'],
      [str(args.vivado_bin/'xelab'),'-mt','off','driftadapt_u55c_250mhz','-s','wrapper_elaboration'],
      [str(args.vivado_bin/'xelab'),'-mt','off','tb','-s','encoded_mlp_tb'],
      [str(args.vivado_bin/'xsim'),'encoded_mlp_tb','-runall']]
    logs=[]
    for i,command in enumerate(commands):
        env={k:v for k,v in os.environ.items() if k not in ('LD_LIBRARY_PATH','PYTHONPATH')}
        result=subprocess.run(command,cwd=build,text=True,capture_output=True,timeout=300,env=env)
        log=result.stdout+result.stderr
        (build/f'step-{i}.log').write_text(log)
        logs.append(log)
        if result.returncode or 'FATAL' in log or 'Fatal:' in log:
            raise RuntimeError(log[-8000:])
    if 'RTL_PASS' not in logs[-1]:
        raise RuntimeError(logs[-1][-8000:])
    print(logs[-1][-1500:])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=ASSETS)
    parser.add_argument('--build',type=Path,default=ASSETS.parent/'build/encoded_mlp_v5/verification')
    parser.add_argument('--vivado-bin',type=Path,default=Path('/home/palhad/Xilinx/Vivado/2022.2/bin'))
    verify(parser.parse_args())
