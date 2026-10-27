"""Regression checks for current checkpoint export and causal hardware updates."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from driftadapt_runtime import ABI, ASSETS, Runtime, fixed_inference, pack_model, write_words


class RuntimeTests(unittest.TestCase):
    def snapshot(self, runtime, version=0):
        start=runtime.next_sample
        ids=runtime.bins[start:start+runtime.size]
        pred,margin=fixed_inference(ids,runtime.images[version])
        current=runtime.method._counts(ids//4,16)
        diff=np.abs(current*runtime.method.reference_samples-runtime.method.reference*runtime.size).sum(axis=1)
        shifted=bool(np.any(diff>2*runtime.config['drift']['threshold']*runtime.size*runtime.method.reference_samples))
        streak=runtime.method.streak+1 if shifted else 0
        return dict(bins=ids,predictions=pred,margins=margin,versions=np.full(runtime.size,version),
                    first_sample=start,window_id=start//runtime.size+1,
                    drift=streak>=runtime.config['drift']['consecutive_windows'])

    def test_export_matches_checkpoint(self):
        runtime=Runtime()
        _,margin=fixed_inference(runtime.bins,runtime.images[0])
        np.testing.assert_array_equal(margin,runtime.method._scores(runtime.data.features[2][:runtime.count].cpu().numpy()))
        self.assertEqual(runtime.images[0].nbytes,1540)

    def test_verified_feedback_produces_usable_image(self):
        runtime=Runtime()
        # Synthetic verified-feedback fixture, never enabled by the controller.
        runtime.config['training']['epochs']=3
        records=[dict(sample_id=i,label=int(runtime.data.labels[2][i]),verified=True,available_window=1)
                 for i in range(32)]
        runtime.feedback.records=records
        runtime.feedback.sample_ids=set(range(32))
        event,candidate=runtime.process(self.snapshot(runtime))
        self.assertTrue(event['adapted'])
        self.assertEqual(runtime.method.feedback_count,32)
        np.testing.assert_array_equal(candidate,pack_model(runtime.method))
        runtime.images[1]=candidate
        runtime.process(self.snapshot(runtime,1))
        self.assertIn(1,runtime.observed_versions)
        self.assertEqual(runtime.method.feedback_count,32)

    def test_invalid_snapshot_is_rejected_before_training(self):
        runtime=Runtime()
        snapshot=self.snapshot(runtime)
        snapshot['margins']=snapshot['margins'].copy()
        snapshot['margins'][0]+=1
        with self.assertRaisesRegex(ValueError,'margin mismatch'):
            runtime.process(snapshot)
        self.assertEqual(runtime.next_sample,0)
        self.assertEqual(runtime.method.version,0)
        snapshot=self.snapshot(runtime)
        snapshot['bins']=snapshot['bins'].copy()
        snapshot['bins'][0,0]=(snapshot['bins'][0,0]+1)%64
        with self.assertRaisesRegex(ValueError,'source bins'):
            runtime.process(snapshot)

    def test_tcl_transport_protocol(self):
        # Exercise the actual Tcl parser/record decoding without a board.
        mock=r"""
set memory [dict create 0 0x44524654 4 0x00050000 8 1 12 19 20 1 24 2 28 1 32 1 36 0 40 1 52 0 56 385 60 2 96 0 100 0 104 385 116 2 124 12345]
set txns [dict create]
proc open_hw_manager {} {}
proc connect_hw_server {args} {}
proc open_hw_target {} {}
proc get_hw_devices {args} {return device}
proc current_hw_device {args} {}
proc refresh_hw_device {args} {}
proc get_hw_axis {args} {return axis}
proc close_hw_target {} {}
proc disconnect_hw_server {} {}
proc close_hw_manager {} {}
proc create_hw_axi_txn {name axis args} {global txns;dict set txns $name $args;return $name}
proc run_hw_axi {name} {
 global txns memory
 set args [dict get $txns $name]
 scan [lindex $args [expr {[lsearch $args -address]+1}]] %x address
 if {[lindex $args [expr {[lsearch $args -type]+1}]] eq "write"} {
   scan [lindex $args [expr {[lsearch $args -data]+1}]] %x value
   dict set memory $address $value
   if {$address==16 && $value==2} {dict set memory 52 1}
 }
}
proc get_property {property obj} {
 global txns memory
 if {$property eq "CELL_NAME"} {return driftadapt_jtag_axi_master}
 set args [dict get $txns $obj]
 scan [lindex $args [expr {[lsearch $args -address]+1}]] %x address
 if {[dict exists $memory $address]} {return [format %08x [dict get $memory $address]]}
 return 00000000
}
proc delete_hw_axi_txn {args} {}
"""
        with tempfile.TemporaryDirectory() as directory:
            image=Path(directory)/'model.mem'
            write_words(image,Runtime().images[0])
            script=Path(__file__).with_name('driftadapt_online_control.tcl').resolve()
            for operation,args in [('status',[]),('snapshot',['auto']),('release',[0,1]),('weights',[str(image)])]:
                command=mock+'\nset argv [list server '+operation+' 12345 '+ ' '.join('{'+str(a)+'}' for a in args)+']\nsource {'+str(script)+'}\n'
                result=subprocess.run(['tclsh'],input=command,text=True,capture_output=True,check=True)
                lines=[line.split('=',1)[1] for line in result.stdout.splitlines() if line.startswith('DRIFTADAPT_JSON=')]
                self.assertEqual(len(lines),1,result.stderr)
                value=json.loads(lines[0])
                self.assertEqual(value['abi'],ABI)
                if operation=='snapshot':
                    self.assertEqual(value['first_sample'],0)
                    self.assertEqual(value['bins'],[[0]*16])
                if operation=='weights':
                    self.assertEqual(value['version'],1)


if __name__=='__main__':
    unittest.main()
