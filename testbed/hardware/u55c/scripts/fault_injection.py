#!/usr/bin/env python3
"""Fault injection on the ABI 5 model-update path (docs/dsn/04-update-path-faults.md).

Run on a freshly programmed board. The last scenario (FI-1) deliberately leaves the
core in an unrecoverable 'updating' state, so reprogram afterwards. Outcomes are
judged from fabric registers (staged_words, version, errors, captured) and from
bit-exact verification of every drained window against the image each flow's
version stamp names. Commits take effect only once the host drains a window
(finding F4), so every commit is followed by settle().

Status word bits: 0-1 window banks full, 3 captured==SAMPLE_COUNT, 4 loaded,
5 updating, 6 any model error, 7 release error.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from driftadapt_runtime import ASSETS, CONFIG, Runtime, fixed_inference, write_words
from run_online_adaptation import Jtag

UPDATING = 1 << 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--assets', type=Path, default=ASSETS)
    parser.add_argument('--vivado', default='/home/palhad/Xilinx/Vivado/2022.2/bin/vivado')
    parser.add_argument('--server', default='TCP:localhost:3121')
    parser.add_argument('--output', type=Path, default=ASSETS.parents[2] / 'results' / 'fault-injection.json')
    parser.add_argument('--skip-fi1', action='store_true', help='do not run the deadlocking scenario')
    args = parser.parse_args()
    runtime = Runtime(args.config, args.assets)
    good_image = runtime.images[0]
    transport = Jtag(args.vivado, args.server, runtime.manifest['build_id'], persistent=True)
    log = []
    images = {0: good_image}          # version -> image the host sent for it
    tmp = Path(tempfile.mkdtemp())
    good = tmp / 'good.mem'
    write_words(good, good_image)
    corrupt_image = good_image.copy()
    corrupt_image[380] = (int(corrupt_image[380]) + 0x00100000) & 0xffffffff   # layer-2 bias of logit 0 -> every margin shifts
    bad = tmp / 'bad.mem'
    write_words(bad, corrupt_image)

    def expect(scenario, name, condition, detail):
        log.append(dict(scenario=scenario, check=name, passed=bool(condition), detail=detail))
        print(f"{scenario:5s} {'PASS' if condition else 'FAIL'}  {name}: {detail}", flush=True)

    def drain(scenario, must_include=None):
        """Snapshot, verify every flow against the image of its stamped version, release."""
        deadline = time.monotonic() + 60
        while transport.command('status')['status'] & 3 == 0:
            if time.monotonic() > deadline:
                raise TimeoutError('no window completed')
            time.sleep(0.1)
        snap = transport.command('snapshot', 'auto')
        ids = np.asarray(snap['bins'], dtype=np.int64)
        versions = np.asarray(snap['versions'])
        ok, seen = True, sorted(set(int(v) for v in versions))
        for v in seen:
            take = versions == v
            if v not in images:
                ok = False
                continue
            pred, margins = fixed_inference(ids[take], images[v])
            ok &= bool(np.array_equal(pred, np.asarray(snap['predictions'])[take]) and
                       np.array_equal(margins, np.asarray(snap['margins'])[take]))
        expect(scenario, f'window {snap["window_id"]} verifies per stamped version {seen}', ok, f"first_sample {snap['first_sample']}")
        if must_include is not None:
            expect(scenario, f'window {snap["window_id"]} contains flows scored by version {must_include}',
                   must_include in seen, f'versions {seen}')
        transport.command('release', snap['bank'], snap['window_id'])
        return snap

    def settle(scenario, expected_version, max_windows=4):
        """Drain windows until the fabric reports expected_version (commit latency, F4)."""
        for _ in range(max_windows):
            if transport.command('status')['version'] >= expected_version:
                break
            drain(scenario)
        s = transport.command('status')
        expect(scenario, f'commit landed as version {expected_version} after draining', s['version'] == expected_version and s['errors'] == 0,
               f"version {s['version']} errors {s['errors']} status {s['status']:#x}")
        return s

    def drain_until(scenario, wanted_version, max_windows=3):
        """The bank already full at swap time was scored before it; the first window with
        flows of the new version arrives one window later. Drain until it appears."""
        for _ in range(max_windows):
            snap = drain(scenario)
            if wanted_version in set(int(v) for v in snap['versions']):
                expect(scenario, f'window {snap["window_id"]} contains flows scored by version {wanted_version}', True,
                       f"versions {sorted(set(int(v) for v in snap['versions']))}")
                return snap
        expect(scenario, f'a window with flows scored by version {wanted_version} appears', False, f'none within {max_windows} windows')
        return snap

    s = transport.command('status')
    expect('pre', 'board healthy before injection', s['errors'] == 0 and s['version'] == 0, f"version {s['version']} errors {s['errors']} captured {s['captured']}")
    version = 0

    # FI-2: staging write outside an update -> refused by the register map, core untouched
    r = transport.command('fi_write', 0x4000, int(good_image[0]))
    expect('FI-2', 'write outside an update is refused: nothing staged, version and errors unchanged',
           r['staged_words'] == 0 and r['version'] == version and r['errors'] == 0 and not (r['status'] & UPDATING),
           f"accepted={r['accepted']} staged {r['staged_words']} version {r['version']} errors {r['errors']} status {r['status']:#x}")
    drain('FI-2')

    # FI-3: out-of-range staging index during an update
    r = transport.command('fi_write', 0x010, 1)
    expect('FI-3', 'update_begin accepted (updating bit set, errors clear)', bool(r['status'] & UPDATING) and r['errors'] == 0, f"status {r['status']:#x} errors {r['errors']}")
    r = transport.command('fi_write', 0x4000 + 4 * 385, 0xdeadbeef)
    expect('FI-3', 'word index 385 is not staged', r['staged_words'] == 0 and r['errors'] == 0, f"staged {r['staged_words']} errors {r['errors']}")

    # FI-4: nested update_begin refused; then a normal commit of the same image
    r = transport.command('fi_stage', str(good))
    expect('FI-4', 'full image staged', r['staged_words'] == 385, f"written {r['written']} staged {r['staged_words']}")
    r = transport.command('fi_write', 0x010, 1)
    expect('FI-4', 'nested update_begin refused: still updating with 385 staged, version unchanged',
           r['staged_words'] == 385 and r['version'] == version and bool(r['status'] & UPDATING), f"staged {r['staged_words']} version {r['version']} status {r['status']:#x}")
    r = transport.command('fi_write', 0x010, 2)
    expect('FI-4', 'commit accepted: pending until the datapath reaches an inference boundary (F4)',
           r['errors'] == 0 and r['version'] in (version, version + 1), f"version {r['version']} status {r['status']:#x}")
    version += 1
    images[version] = good_image
    settle('FI-4', version)
    drain_until('FI-4', version)

    # FI-5: corrupted image passes the fabric, is observable through verification, and is rolled back
    r = transport.command('weights', str(bad))
    version += 1
    images[version] = corrupt_image
    expect('FI-5', 'corrupted image accepted by the fabric (no integrity check)', r['errors'] == 0 if 'errors' in r else True, f"reported version {r['version']} committed {r['committed']}")
    settle('FI-5', version)
    snap = drain_until('FI-5', version)
    take = np.asarray(snap['versions']) == version
    pred_good, margin_good = fixed_inference(np.asarray(snap['bins'], dtype=np.int64)[take], good_image)
    expect('FI-5', 'flows scored by the corrupted version do NOT match the correct image (corruption is observable)',
           not np.array_equal(margin_good, np.asarray(snap['margins'])[take]), f"{int(take.sum())} flows checked")
    r = transport.command('weights', str(good))
    version += 1
    images[version] = good_image
    settle('FI-5', version)
    drain_until('FI-5', version)
    expect('FI-5', 'rollback is an ordinary versioned commit of the previous image bytes',
           np.array_equal(images[version], images[version - 2]), f"version {version} == bytes of version {version - 2}")

    # FI-1: incomplete staging then commit -> no abort path, datapath halts
    if not args.skip_fi1:
        transport.command('fi_write', 0x010, 1)
        r = transport.command('fi_stage', str(good), 384)
        expect('FI-1', '384 of 385 words staged', r['staged_words'] == 384, f"staged {r['staged_words']}")
        r = transport.command('fi_write', 0x010, 2)
        expect('FI-1', 'commit refused: version unchanged, core still updating',
               r['version'] == version and bool(r['status'] & UPDATING), f"version {r['version']} status {r['status']:#x}")
        for _ in range(2):
            if transport.command('status')['status'] & 3:
                snap = transport.command('snapshot', 'auto')
                transport.command('release', snap['bank'], snap['window_id'])
        c0 = transport.command('status')['captured']
        time.sleep(3)
        s = transport.command('status')
        expect('FI-1', 'inference halted: captured does not advance while the update is abandoned (finding F1)',
               s['captured'] == c0 and bool(s['status'] & UPDATING), f"captured {c0} -> {s['captured']} over 3 s, status {s['status']:#x}; reprogram required")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(build_id=runtime.manifest['build_id'], checks=log), indent=2) + '\n')
    print(f"{sum(c['passed'] for c in log)}/{len(log)} checks passed; log at {args.output}")
    transport.close()


if __name__ == '__main__':
    main()
