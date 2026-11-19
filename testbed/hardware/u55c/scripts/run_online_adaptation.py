#!/usr/bin/env python3
"""Run only current DriftAdapt on software replay or ABI 5 U55C hardware."""
import argparse
import atexit
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import queue
import threading
import tempfile
import time
import numpy as np
from driftadapt_runtime import ABI, ASSETS, CONFIG, Runtime, fixed_inference, write_words


class Jtag:
    def __init__(self, vivado, server, build_id, persistent=False):
        self.vivado, self.server, self.build_id = vivado, server, build_id
        self.persistent = persistent
        self.process = None
        self.responses = queue.Queue()
        self.log = deque(maxlen=80)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.write('quit\n')
                self.process.stdin.flush()
                self.process.wait(timeout=15)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=15)

    def _persistent_command(self, operation, args):
        if self.process is None:
            env = {k:v for k,v in os.environ.items() if k not in ('LD_LIBRARY_PATH','PYTHONPATH')}
            self.process = subprocess.Popen([self.vivado, '-mode','batch','-nojournal','-nolog','-notrace',
                '-source',str(Path(__file__).with_name('driftadapt_online_control.tcl')),
                '-tclargs',self.server,'serve',str(self.build_id)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env)
            def collect():
                for line in self.process.stdout:
                    self.log.append(line)
                    if line.startswith(('DRIFTADAPT_JSON=', 'DRIFTADAPT_ERROR=')):
                        self.responses.put(line.strip())
                self.responses.put(None)
            threading.Thread(target=collect,daemon=True).start()
            atexit.register(self.close)
        fields = [operation, *map(str,args)]
        if any('\t' in field or '\n' in field for field in fields):
            raise ValueError('Invalid JTAG command field')
        request = '\t'.join(fields).encode().hex()
        self.process.stdin.write(request+'\n')
        self.process.stdin.flush()
        try:
            response = self.responses.get(timeout=300)
        except queue.Empty:
            self.close()
            raise TimeoutError('JTAG operation timed out: '+''.join(self.log)[-6000:])
        if response is None:
            raise RuntimeError('JTAG session ended: '+''.join(self.log)[-6000:])
        if response.startswith('DRIFTADAPT_ERROR='):
            raise RuntimeError(bytes.fromhex(response.split('=',1)[1]).decode())
        return json.loads(response.split('=',1)[1])

    def command(self, operation, *args):
        if self.persistent:
            value = self._persistent_command(operation,args)
            if value['abi'] != ABI or value['build_id'] != self.build_id:
                raise ValueError('FPGA ABI/build mismatch')
            return value
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
    transport = Jtag(args.vivado, args.server, runtime.manifest['build_id'], persistent=True) if args.backend=='jtag' else None
    version = 0
    if transport:
        initial = transport.command('status')
        if initial['version'] != 0 or initial['errors'] or initial['sample_count'] != runtime.count:
            raise ValueError('Start with a freshly reset, matching FPGA image')
    events = []
    idle_since = time.monotonic()
    # A commit takes effect at the next inference boundary, which the core only
    # reaches once the host frees a window bank. So a commit normally stays
    # pending across one loop iteration; the new image is registered under the
    # version it will become, and further candidates queue until it lands.
    expected = None
    queued = deque()
    commit_log = []
    pending_images = []   # software backend: (activation flow index, version)
    active_version = 0    # software backend: image currently scoring flows
    schedule = None
    if args.commit_schedule:
        # Replay the exact activation points recorded by a board run: for each
        # version, the first flow index its image scored (from model_versions_scored).
        board = json.load(args.commit_schedule.open())
        schedule = {}
        for event in board['events']:
            start = (event['window_id'] - 1) * runtime.size
            offset = 0
            for v, n in sorted(((int(v), n) for v, n in event['model_versions_scored'].items())):
                schedule.setdefault(v, start + offset)
                offset += n
        schedule.pop(0, None)

    def send(candidate):
        nonlocal version
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.mem'
            write_words(path, candidate)
            update = transport.command('weights', path)
        version += 1
        if update['version'] not in (version-1, version):
            raise ValueError('FPGA model version out of sync with the host commit count')
        runtime.images[version] = candidate
        commit_log.append(dict(version=version, window_id=runtime.next_sample//runtime.size,
                               committed_immediately=bool(update['committed'])))
        return None if update['committed'] else version

    while runtime.next_sample < runtime.count:
        if transport:
            status = transport.command('status')
            if status['errors']:
                raise ValueError('FPGA reported an error')
            if expected is not None and status['version'] >= expected:
                expected = None
            if expected is None and queued:
                expected = send(queued.popleft())
            if not status['status'] & 3:
                if time.monotonic()-idle_since > args.timeout:
                    raise TimeoutError('FPGA did not produce the next window')
                time.sleep(0.1)
                continue
            snapshot = transport.command('snapshot','auto')
        else:
            start = runtime.next_sample
            ids = runtime.bins[start:start+runtime.size]
            # Model the board's commit latency: an image sent after window w becomes
            # active --commit-lag flows after the end of that window (0 = immediate).
            flow_versions = np.full(runtime.size, active_version)
            for activation, pending_version in sorted(pending_images):
                if activation < start + runtime.size:
                    flow_versions[max(0, activation - start):] = pending_version
                    active_version = pending_version
                    pending_images.remove((activation, pending_version))
            predictions = np.empty(runtime.size, dtype=np.int64)
            margins = np.empty(runtime.size, dtype=np.int32)
            for v in np.unique(flow_versions):
                take = flow_versions == v
                predictions[take], margins[take] = fixed_inference(ids[take], runtime.images[int(v)])
            # Independent hardware-TV integer expression, no labels.
            current = runtime.method._counts(ids//4, 16)
            diff = np.abs(current*runtime.method.reference_samples -
                          runtime.method.reference*runtime.size).sum(axis=1)
            shifted = bool(np.any(diff > 2*runtime.config['drift']['threshold']*
                                  runtime.size*runtime.method.reference_samples))
            streak = runtime.method.streak+1 if shifted else 0
            snapshot = dict(bins=ids, predictions=predictions, margins=margins,
                versions=flow_versions,first_sample=start,
                window_id=start//runtime.size+1,
                drift=streak>=runtime.config['drift']['consecutive_windows'])
        event, candidate = runtime.process(snapshot)
        # Free space before waiting for an inference-boundary commit. Both
        # windows can be full with another result blocked at the core output.
        if transport:
            transport.command('release',snapshot['bank'],snapshot['window_id'])
        if candidate is not None:
            if transport:
                queued.append(candidate)
                if expected is None:
                    expected = send(queued.popleft())
            else:
                version += 1
                runtime.images[version] = candidate
                if schedule is None and not args.commit_lag:
                    active_version = version
                elif schedule is not None:
                    # A version absent from the schedule never scored a flow on the board
                    # (committed after the replay finished): activate it past the end.
                    pending_images.append((schedule.get(version, runtime.count), version))
                elif args.commit_lag:
                    pending_images.append((runtime.next_sample + args.commit_lag, version))
        event['model_versions_scored'] = {int(v): int(n) for v, n in
                                          zip(*np.unique(np.asarray(snapshot['versions']), return_counts=True))}
        events.append(dict(window_id=snapshot['window_id'], **event))
        if transport:
            print(f'FPGA progress: {runtime.next_sample}/{runtime.count} validated flows', flush=True)
        idle_since = time.monotonic()
    report = dict(backend=args.backend, architecture=runtime.method.architecture,
        hardware_verified=transport is not None, complete=True,
        unobserved_model_versions=sorted(set(runtime.images)-runtime.observed_versions),
        metrics=runtime.report(), events=events)
    if transport:
        report['commits'] = commit_log
        report['commit_pending_at_end'] = expected
        report['candidates_not_sent'] = len(queued)
        report['final_fpga_status'] = transport.command('status')
        if report['final_fpga_status']['errors'] or report['final_fpga_status']['captured'] != runtime.count:
            raise ValueError('Incomplete FPGA run')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='events'},indent=2))
    if transport:
        transport.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('jtag','software'),default='jtag')
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--assets',type=Path,default=ASSETS)
    parser.add_argument('--vivado',default='/home/palhad/Xilinx/Vivado/2022.2/bin/vivado')
    parser.add_argument('--server',default='TCP:localhost:3121')
    parser.add_argument('--timeout',type=float,default=60)
    parser.add_argument('--commit-lag',type=int,default=0,
        help='software backend: flows after the end of a window before its committed image is active')
    parser.add_argument('--commit-schedule',type=Path,
        help='software backend: a board report whose per-window model_versions_scored fix each version\'s activation flow')
    parser.add_argument('--output',type=Path,default=ASSETS.parents[2]/'results/encoded-mlp.json')
    run(parser.parse_args())
