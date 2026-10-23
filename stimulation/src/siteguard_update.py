"""Verified, delayed feedback and hinge-active Euclidean mirror descent."""
from collections import deque
from pathlib import Path
import json
import math
import numpy as np


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
