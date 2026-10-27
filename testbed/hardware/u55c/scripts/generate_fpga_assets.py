#!/usr/bin/env python3
"""Export the current simulation DriftAdapt checkpoint for the U55C encoded MLP."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
from driftadapt_runtime import (ABI, ASSETS, CONFIG, MODEL_WORDS, fixed_inference,
                               load_model, pack_model, sha256, write_words)


def generate(config_path, output, checkpoint=None):
    config, data, method = load_model(config_path, checkpoint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    size = config['stream']['window_size']
    count = len(data.features[2]) // size * size
    if not 1 <= size <= 1024 or not count:
        raise ValueError('Hardware window size must be 1..1024 with at least one complete window')
    for part in data.features:
        raw = part.cpu().numpy()
        if not np.array_equal(method._encoder_indices(raw) // 4, method._indices(raw)):
            raise ValueError('Hardware monitor requires aligned source encoder and histogram ranges')
    bins = method._encoder_indices(data.features[2][:count].cpu().numpy())
    reference = method._counts(method._indices(data.features[0].cpu().numpy()), method.bins)
    words = pack_model(method)
    predictions, scores = fixed_inference(bins, words)
    if not np.array_equal(scores, method._scores(data.features[2][:count].cpu().numpy())):
        raise ValueError('Integer export does not match simulation')
    write_words(output / 'driftadapt_weights.mem', words)
    write_words(output / 'driftadapt_reference.mem', reference.ravel())
    (output / 'driftadapt_samples.mem').write_text(''.join(
        row.astype('u1').tobytes()[::-1].hex() + '\n' for row in bins))
    fraction = Fraction(str(config['drift']['threshold']))
    params = dict(SAMPLE_COUNT=count, WINDOW_SIZE=size, MODEL_WORDS=MODEL_WORDS,
                  REFERENCE_SAMPLES=len(data.features[0]), TV_NUM=fraction.numerator,
                  TV_DEN=fraction.denominator, CONSECUTIVE=config['drift']['consecutive_windows'])
    files = {name: sha256(output / name) for name in
             ('driftadapt_weights.mem', 'driftadapt_samples.mem', 'driftadapt_reference.mem')}
    build_id = int(hashlib.sha256(json.dumps([params, files], sort_keys=True).encode()).hexdigest()[:8], 16)
    header = ''.join(f'`define DA_{key} {value}\n' for key, value in params.items())
    header += f"`define DA_BUILD_ID 32'h{build_id:08x}\n"
    (output / 'driftadapt_config.vh').write_text(header)
    files['driftadapt_config.vh'] = sha256(output / 'driftadapt_config.vh')
    labels = method._labels(data.labels[2][:count].cpu().numpy())
    manifest = dict(abi=ABI, architecture=method.architecture, layers=[16,16,8,2],
        model_words=MODEL_WORDS, model_payload_bytes=4*MODEL_WORDS,
        inference_parameter_bytes=method.parameter_bytes, sample_count=count, window_size=size,
        preprocessing_id=data.preprocessing_id, drift=config['drift'], build_id=build_id,
        checkpoint=str(method.checkpoint), checkpoint_sha256=sha256(method.checkpoint),
        config=str(Path(config_path).resolve()), files=files,
        input_representation='16 uint8 encoder bin IDs, 0..63; host source-only preprocessing',
        expected=dict(true_positive=int(((predictions==1)&(labels==1)).sum()),
                      true_negative=int(((predictions==0)&(labels==0)).sum()),
                      false_positive=int(((predictions==1)&(labels==0)).sum()),
                      false_negative=int(((predictions==0)&(labels==1)).sum())))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-directory', type=Path, default=ASSETS)
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.output_directory, args.checkpoint), indent=2))
