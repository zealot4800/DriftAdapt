#!/usr/bin/env python3
"""Run only current DriftAdapt on software replay or ABI 5 U55C hardware."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
from driftadapt_runtime import ABI, ASSETS, CONFIG, Runtime, fixed_inference, write_words


class Jtag:
    def __init__(self, vivado, server, build_id):
        self.vivado, self.server, self.build_id = vivado, server, build_id

    def command(self, operation, *args):
        env = {k:v for k,v in os.environ.items() if k not in ('LD_LIBRARY_PATH','PYTHONPATH')}
        result = subprocess.run([self.vivado, '-mode','batch','-nojournal','-nolog','-notrace',
            '-source',str(Path(__file__).with_name('driftadapt_online_control.tcl')),
            '-tclargs', self.server, operation, str(self.build_id), *map(str,args)],
            text=True, capture_output=True, env=env, timeout=300)
        records = [line.split('=',1)[1] for line in result.stdout.splitlines()
                   if line.startswith('DRIFTADAPT_JSON=')]
        if result.returncode or len(records) != 1:
            raise RuntimeError('JTAG operation failed: '+(result.stdout+result.stderr)[-6000:])
        value = json.loads(records[0])
        if value['abi'] != ABI or value['build_id'] != self.build_id:
            raise ValueError('FPGA ABI/build mismatch')
        return value


def run(args):
    runtime = Runtime(args.config, args.assets, args.checkpoint)
    transport = Jtag(args.vivado, args.server, runtime.manifest['build_id']) if args.backend=='jtag' else None
    version = 0
    if transport:
        initial = transport.command('status')
        if initial['version'] != 0 or initial['errors'] or initial['sample_count'] != runtime.count:
            raise ValueError('Start with a freshly reset, matching FPGA image')
    events = []
    idle_since = time.monotonic()
    while runtime.next_sample < runtime.count:
        if transport:
            status = transport.command('status')
            if status['errors']:
                raise ValueError('FPGA reported an error')
            if not status['status'] & 3:
                if time.monotonic()-idle_since > args.timeout:
                    raise TimeoutError('FPGA did not produce the next window')
                time.sleep(0.1)
                continue
            snapshot = transport.command('snapshot','auto')
        else:
            start = runtime.next_sample
            ids = runtime.bins[start:start+runtime.size]
            predictions, margins = fixed_inference(ids, runtime.images[version])
            # Independent hardware-TV integer expression, no labels.
            current = runtime.method._counts(ids//4, 16)
            diff = np.abs(current*runtime.method.reference_samples -
                          runtime.method.reference*runtime.size).sum(axis=1)
            shifted = bool(np.any(diff > 2*runtime.config['drift']['threshold']*
                                  runtime.size*runtime.method.reference_samples))
            streak = runtime.method.streak+1 if shifted else 0
            snapshot = dict(bins=ids, predictions=predictions, margins=margins,
                versions=np.full(runtime.size,version),first_sample=start,
                window_id=start//runtime.size+1,
                drift=streak>=runtime.config['drift']['consecutive_windows'])
        event, candidate = runtime.process(snapshot)
        # Free space before waiting for an inference-boundary commit. Both
        # windows can be full with another result blocked at the core output.
        if transport:
            transport.command('release',snapshot['bank'],snapshot['window_id'])
        if candidate is not None:
            if transport:
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/'model.mem'
                    write_words(path,candidate)
                    update = transport.command('weights',path)
                version = update['version']
            else:
                version += 1
            runtime.images[version] = candidate
        events.append(dict(window_id=snapshot['window_id'], **event))
        idle_since = time.monotonic()
    report = dict(backend=args.backend, architecture=runtime.method.architecture,
        hardware_verified=transport is not None, complete=True,
        unobserved_model_versions=sorted(set(runtime.images)-runtime.observed_versions),
        metrics=runtime.report(), events=events)
    if transport:
        report['final_fpga_status'] = transport.command('status')
        if report['final_fpga_status']['errors'] or report['final_fpga_status']['captured'] != runtime.count:
            raise ValueError('Incomplete FPGA run')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='events'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('jtag','software'),default='jtag')
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--assets',type=Path,default=ASSETS)
    parser.add_argument('--vivado',default='/home/palhad/Xilinx/Vivado/2022.2/bin/vivado')
    parser.add_argument('--server',default='TCP:localhost:3121')
    parser.add_argument('--timeout',type=float,default=60)
    parser.add_argument('--output',type=Path,default=ASSETS.parents[2]/'results/encoded-mlp.json')
    run(parser.parse_args())
