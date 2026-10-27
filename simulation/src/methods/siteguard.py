"""SiteGuard v2: flow RBF encoding, one-class detection and verified updates."""
from pathlib import Path
from collections import deque
import json
import math
import numpy as np
import torch
from ..models import OneClassDetector, SourceScaler


class OnlineUpdater:
    def __init__(self, detector, config, checkpoint_dir):
        self.detector = detector
        self.config = dict(config)
        self.directory = Path(checkpoint_dir)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.feedback = deque(maxlen=config['feedback_window'])
        self.seen = set()
        self.history = []
        self.iteration = 0
        self.version = 0
        self.last_available = -float('inf')

    def observe(self, *, window_id, vector, prediction, window_end, available_at,
                now, label, verified):
        if verified is not True or type(label) is not int or label not in (-1, 1):
            raise ValueError('updates require explicitly verified -1/+1 feedback')
        if window_id in self.seen:
            raise ValueError('duplicate feedback')
        if not all(math.isfinite(v) for v in (window_end, available_at, now)):
            raise ValueError('feedback timestamps must be finite')
        if available_at < window_end or available_at > now or available_at < self.last_available:
            raise ValueError('feedback must be chronologically available after prediction')
        if prediction not in (-1, 1):
            raise ValueError('invalid original prediction')
        vector = np.asarray(vector, dtype=float)
        score = float(self.detector.score(vector))
        self.seen.add(window_id)
        self.last_available = available_at
        self.feedback.append((prediction, label))
        benign = [p for p, y in self.feedback if y == -1]
        alpha = sum(p == 1 for p in benign) / len(benign) if benign else None
        active = (len(benign) >= self.config['minimum_benign_feedback'] and
                  abs(alpha - self.config['beta']) > self.config['delta'])
        margin = label * score
        changed = False
        eta = 0.0
        delta_norm = 0.0
        if active:
            self.iteration += 1
            eta = self.config['learning_rate']
            if self.config['schedule'] == 'inverse_sqrt':
                eta /= math.sqrt(self.iteration)
            if margin < 1.0:
                delta = eta * label * vector
                candidate = self.detector.weights + delta
                if not np.isfinite(candidate).all():
                    raise ValueError('nonfinite online update')
                delta_norm = float(np.linalg.norm(delta))
                changed = delta_norm > 0
                if changed:
                    # Store both sides, not merely a norm, for exact audit and rollback.
                    np.savez(self.directory / f'update-{self.version + 1:06d}.npz',
                        before=self.detector.weights, after=candidate,
                        rho=np.asarray(self.detector.rho), delta=delta)
                    self.detector.weights = candidate
                    self.version += 1
        record = {'window_id': window_id, 'available_at': available_at,
                  'verified_label': label, 'original_prediction': prediction,
                  'false_alarm_rate': alpha, 'verified_benign_count': len(benign),
                  'update_phase': active, 'hinge_loss_before': max(0.0, 1 - margin),
                  'learning_rate': eta, 'parameter_delta_norm': delta_norm,
                  'changed': changed, 'model_version': self.version,
                  'rho': self.detector.rho}
        self.history.append(record)
        with (self.directory / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        return record

    def rollback(self, update_version):
        path = self.directory / f'update-{update_version:06d}.npz'
        with np.load(path, allow_pickle=False) as saved:
            before = saved['before']
            if before.shape != self.detector.weights.shape:
                raise ValueError('rollback checkpoint dimension mismatch')
            self.detector.weights = before.copy()
            self.detector.rho = float(saved['rho'])
        # Version counter stays monotonic; rollback is an explicit logged event.
        self.version += 1
        with (self.directory / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps({'rollback_before_update': update_version,
                                     'model_version': self.version}) + '\n')


class SiteGuard:
    """Benign-trained RBF flow detector with verified false-alarm-driven updates."""
    unit = 'flow'
    representation = 'siteguard_flow_rff_v2'
    map_size = 256
    evaluation_target = 'anomaly'

    def __init__(self, config, output):
        from ..data import benign_classes
        from ..config import path_for
        self.config = config
        self.output = Path(output)
        if config['siteguard'].get('feedback'):
            raise ValueError('Flow feedback uses feedback.file with sample_id and available_window; only shared flow feedback is supported')
        self.benign = benign_classes(config, required=True)
        self.checkpoint = path_for(config, config['siteguard']['checkpoint'])
        self.classifier_config = config['siteguard']['classifier']
        if self.classifier_config['objective'] != 'slack_corrected_linear':
            raise ValueError('Flow SiteGuard retains the slack-corrected linear one-class objective')
        self.detector = OneClassDetector(self.classifier_config)
        self.scaler = SourceScaler()
        self.map_weights = self.map_offset = None
        self.pending = {}
        self.updater = None
        self.update_bytes = self.feedback_count = 0

    def _fit_map(self, features):
        variance = max(float(features.var()), np.finfo(np.float64).eps)
        gamma = 1.0 / (features.shape[1] * variance)
        rng = np.random.default_rng(self.config['runtime']['seed'])
        self.map_weights = rng.normal(0.0, np.sqrt(2.0 * gamma),
                                      (features.shape[1], self.map_size))
        self.map_offset = rng.uniform(0.0, 2.0 * np.pi, self.map_size)
        return gamma

    def _map(self, features):
        return np.sqrt(2.0 / self.map_size) * np.cos(
            np.asarray(features) @ self.map_weights + self.map_offset)

    def _load(self, data):
        saved = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        if (saved.get('representation') != self.representation
            or saved.get('preprocessing_id') != data.preprocessing_id
            or saved.get('benign_classes') != self.benign
            or saved.get('classifier') != self.classifier_config):
            raise ValueError('SiteGuard checkpoint representation, source preprocessing, benign classes, or classifier differ; train a new checkpoint')
        self.scaler.minimum = saved['minimum'].numpy()
        self.scaler.span = saved['span'].numpy()
        self.map_weights = saved['map_weights'].numpy()
        self.map_offset = saved['map_offset'].numpy()
        self.detector.weights = saved['weights'].numpy().copy()
        self.detector.rho = saved['rho']
        return saved['training']

    def train(self, data):
        if self.checkpoint.exists():
            return dict(self._load(data), reused_checkpoint=str(self.checkpoint))
        labels = data.labels[0].cpu().numpy()
        benign = np.isin(labels, self.benign)
        features = data.features[0].cpu().numpy()[benign]
        if len(features) < 2:
            raise ValueError(f'Flow SiteGuard needs at least two benign training flows in the ground-truth label column; benign classes {self.benign}, found {len(features)}')
        self.scaler.fit(features)
        scaled = self.scaler.transform(features)
        gamma = self._fit_map(scaled)
        self.detector.fit_benign(self._map(scaled))
        report = {'representation':self.representation, 'benign_training_flows':len(features),
                  'excluded_nonbenign_training_flows':int((~benign).sum()),
                  'packet_sequences_used':False, 'dft_used':False,
                  'flow_map':'random_fourier_rbf', 'flow_map_size':self.map_size,
                  'flow_map_gamma':gamma}
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'representation':self.representation, 'preprocessing_id':data.preprocessing_id,
                    'benign_classes':self.benign, 'classifier':self.classifier_config,
                    'minimum':torch.from_numpy(self.scaler.minimum), 'span':torch.from_numpy(self.scaler.span),
                    'map_weights':torch.from_numpy(self.map_weights), 'map_offset':torch.from_numpy(self.map_offset),
                    'weights':torch.from_numpy(self.detector.weights), 'rho':self.detector.rho,
                    'training':report}, self.checkpoint)
        return report

    def prepare(self, data):
        self._load(data)
        self.updater = OnlineUpdater(self.detector, self.config['siteguard']['online'], self.output/'updates')

    def infer(self, observation):
        scaled = self.scaler.transform(observation['features'].cpu().numpy())
        self.vectors = self._map(scaled)
        return (self.detector.predict(self.vectors) == 1).astype(np.int64)

    def adapt(self, observation, prediction):
        before = self.updater.version
        size = self.update_bytes
        for offset, sample_id in enumerate(observation['sample_ids']):
            if sample_id in observation['feedback_sample_ids']:
                self.pending[sample_id] = (self.vectors[offset].copy(), int(prediction[offset]), observation['id'])
        for record in observation['feedback']:
            vector, original, end = self.pending.pop(record['sample_id'])
            result = self.updater.observe(window_id=str(record['sample_id']), vector=vector,
                prediction=1 if original else -1, window_end=end, available_at=record['available_window'],
                now=observation['id'], label=-1 if record['label'] in self.benign else 1, verified=True)
            self.feedback_count += 1
            if result['changed']:
                self.update_bytes += self.detector.weights.nbytes
        return {'adapted':self.updater.version > before, 'model_update_bytes':self.update_bytes-size,
                'verified_feedback_applied':len(observation['feedback'])}

    def state(self):
        return {'representation':self.representation, 'benign_classes':self.benign,
                'adaptation_count':self.updater.version if self.updater else 0,
                'labeling_cost_samples':self.feedback_count, 'labeling_seconds':None,
                'model_update_bytes':self.update_bytes}
