"""Shared ABI and exact integer inference for the encoded MLP hardware testbed."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from simulation.src.config import load_config
from simulation.src.data import FlowData, ReleasedFeedback
from simulation.src.methods.driftadapt import DriftAdapt

ABI = 0x00050000
MODEL_WORDS = 385
ASSETS = Path(__file__).resolve().parents[1] / 'assets'
CONFIG = ROOT / 'simulation/configs/cic-ids2017.yaml'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(config_path=CONFIG, checkpoint=None):
    config = load_config(config_path)
    torch.set_num_threads(config['runtime'].get('threads', 1))
    torch.manual_seed(config['runtime']['seed'])
    data = FlowData(config)
    method = DriftAdapt(config, Path(config['_base']))
    if checkpoint:
        method.checkpoint = Path(checkpoint)
    method.prepare(data)
    if len(data.feature_names) != 16 or method.encoder_bins != 64 or method.bins != 16:
        raise ValueError('U55C image requires 16 features, 64 encoder bins and 16 monitor bins')
    return config, data, method


def pack_model(method):
    payload = method.input_table.astype('i1').tobytes()
    payload += b''.join(w.astype('i1').tobytes() for w, _, _ in method.packed)
    payload += b''.join(b.astype('<i4').tobytes() for _, b, _ in method.packed)
    payload += np.asarray([m for _, _, m in method.packed[:2]], dtype='<u4').tobytes()
    payload += np.asarray([method.threshold], dtype='<i4').tobytes()
    words = np.frombuffer(payload, dtype='<u4').copy()
    assert len(words) == MODEL_WORDS
    return words


def fixed_inference(bins, words):
    bins = np.asarray(bins, dtype=np.int64)
    if bins.ndim != 2 or bins.shape[1] != 16 or np.any((bins < 0) | (bins > 63)):
        raise ValueError('Invalid encoder bin IDs')
    raw = np.asarray(words, dtype='<u4').tobytes()
    table = np.frombuffer(raw, dtype='i1', count=1024).reshape(16, 64)
    weights = np.frombuffer(raw, dtype='i1', count=400, offset=1024)
    biases = np.frombuffer(raw, dtype='<i4', count=26, offset=1424)
    multipliers = np.frombuffer(raw, dtype='<u4', count=2, offset=1528)
    threshold = int(np.frombuffer(raw, dtype='<i4', count=1, offset=1536)[0])
    x = table[np.arange(16), bins].astype(np.int64)
    wi = bi = 0
    for layer, (nin, nout) in enumerate(((16, 16), (16, 8), (8, 2))):
        x = x @ weights[wi:wi+nin*nout].reshape(nout, nin).astype(np.int64).T + biases[bi:bi+nout]
        if np.any((x < -2**31) | (x >= 2**31)):
            raise ValueError('Accumulator overflow')
        wi += nin*nout
        bi += nout
        if layer < 2:
            x = np.clip((x.clip(0) * int(multipliers[layer]) + 2**23) >> 24, 0, 255)
    margin = x[:, 1] - x[:, 0]
    if np.any((margin < -2**31) | (margin >= 2**31)):
        raise ValueError('Decision margin overflow')
    return (margin >= threshold).astype(np.int64), margin.astype(np.int32)


def read_words(path):
    return np.asarray([int(line, 16) for line in Path(path).read_text().splitlines()], dtype=np.uint32)


def write_words(path, words):
    Path(path).write_text(''.join(f'{int(w) & 0xffffffff:08x}\n' for w in words))


def load_assets(directory=ASSETS):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['abi'] != ABI or manifest['model_words'] != MODEL_WORDS:
        raise ValueError('Obsolete hardware assets: regenerate the encoded MLP image')
    for name, digest in manifest['files'].items():
        if sha256(directory / name) != digest:
            raise ValueError(f'Asset checksum mismatch: {name}')
    return manifest


class Runtime:
    """Only explicit verified feedback can update the model; evaluation is separate."""
    def __init__(self, config_path=CONFIG, assets=ASSETS, checkpoint=None):
        self.config, self.data, self.method = load_model(config_path, checkpoint)
        self.manifest = load_assets(assets)
        self.size = self.config['stream']['window_size']
        self.count = len(self.data.features[2]) // self.size * self.size
        if (self.manifest['preprocessing_id'] != self.data.preprocessing_id
                or self.manifest['sample_count'] != self.count
                or self.manifest['window_size'] != self.size
                or self.manifest['drift'] != self.config['drift']):
            raise ValueError('Runtime configuration differs from built assets')
        words = read_words(Path(assets) / 'driftadapt_weights.mem')
        if not np.array_equal(words, pack_model(self.method)):
            raise ValueError('Runtime checkpoint differs from FPGA initial model')
        self.images = {0: words}
        self.observed_versions = set()
        self.bins = self.method._encoder_indices(self.data.features[2][:self.count].cpu().numpy())
        self.feedback = ReleasedFeedback(self.config, self.count)
        self.next_sample = 0
        self.truth, self.predicted = [], []

    def process(self, snapshot):
        start = self.next_sample
        stop = start + self.size
        ids = np.asarray(snapshot['bins'], dtype=np.int64)
        if (snapshot['first_sample'] != start or snapshot['window_id'] != start // self.size + 1
                or stop > self.count or not np.array_equal(ids, self.bins[start:stop])):
            raise ValueError('Window chronology or source bins differ from the exported evaluation partition')
        pred = np.asarray(snapshot['predictions'])
        margins = np.asarray(snapshot['margins'])
        versions = np.asarray(snapshot['versions'])
        if any(v.shape != (self.size,) for v in (pred, margins, versions)):
            raise ValueError('Incomplete window snapshot')
        for version in np.unique(versions):
            if int(version) not in self.images:
                raise ValueError('Unknown FPGA model version')
            take = versions == version
            expected, scores = fixed_inference(ids[take], self.images[int(version)])
            if not np.array_equal(pred[take], expected) or not np.array_equal(margins[take], scores):
                raise ValueError('FPGA prediction/margin mismatch')
            self.observed_versions.add(int(version))
        # Original flow features remain on the host. No packet sequence is reconstructed.
        observation = {'features': self.data.features[2][start:stop],
                       'sample_ids': list(range(start, stop)),
                       'feedback_sample_ids': self.feedback.sample_ids,
                       'feedback': self.feedback.release(start // self.size + 1)}
        self.method.infer(observation)
        event = self.method.adapt(observation, pred)
        if bool(snapshot['drift']) != bool(event['feature_drift_detected']):
            raise ValueError('FPGA histogram trigger differs from simulation')
        candidate = pack_model(self.method) if event['adapted'] else None
        self.truth.extend(self.method._labels(self.data.labels[2][start:stop].cpu().numpy()).tolist())
        self.predicted.extend(pred.tolist())
        self.next_sample = stop
        return event, candidate

    def report(self):
        y, p = np.asarray(self.truth), np.asarray(self.predicted)
        tp = int(((y == 1) & (p == 1)).sum())
        tn = int(((y == 0) & (p == 0)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        return dict(samples=len(y), true_positive=tp, true_negative=tn,
                    false_positive=fp, false_negative=fn,
                    anomaly_f1=2*tp/max(2*tp+fp+fn, 1),
                    false_positive_rate=fp/max(tn+fp, 1),
                    recall=tp/max(tp+fn, 1), adaptation_count=self.method.version,
                    verified_feedback=self.method.feedback_count)
