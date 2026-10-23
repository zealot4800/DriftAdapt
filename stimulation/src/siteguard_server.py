"""Optional software NOC: aligned agents, victim-keyed fusion, independent model."""
import numpy as np
from .siteguard_models import OneClassDetector, SourceScaler


def aggregate(summaries, victim_keys, digest_fields=('packets', 'bytes')):
    if not summaries or len({s['window_id'] for s in summaries}) != 1:
        raise ValueError('summaries must refer to the same detection window')
    if len({s['agent_id'] for s in summaries}) != len(summaries):
        raise ValueError('duplicate agent summary')
    if len(set(victim_keys)) != len(victim_keys):
        raise ValueError('victim vocabulary must be unique')
    if tuple(digest_fields) != ('packets', 'bytes'):
        raise ValueError('only packet/byte intensity digests are implemented')
    stats = np.asarray([s['statistics'] for s in summaries], dtype=float)
    if stats.shape != (len(summaries), 80) or not np.isfinite(stats).all():
        raise ValueError('expected finite 80-dimensional agent statistics')
    stats = stats.reshape(-1, 20, 4)
    merged = stats.max(axis=0)
    merged[:, 0] = stats[:, :, 0].sum(axis=0)
    digest = np.zeros((len(victim_keys) + 1, len(digest_fields)))
    positions = {key: i for i, key in enumerate(victim_keys)}
    for summary in summaries:
        for victim, counts in summary['digest'].items():
            values = np.asarray([counts[name] for name in digest_fields], dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('invalid victim digest')
            digest[positions.get(victim, len(victim_keys))] += values
    return np.concatenate([merged.reshape(-1), digest.reshape(-1)])


class SiteGuardServer:
    def __init__(self, config, victim_keys):
        self.victim_keys = tuple(victim_keys)
        self.fields = tuple(config['server']['digest_fields'])
        self.detector = OneClassDetector(config['classifier'])
        self.scaler = SourceScaler()

    def fit_benign(self, source_summary_windows):
        vectors = np.stack([aggregate(s, self.victim_keys, self.fields) for s in source_summary_windows])
        self.scaler.fit(vectors)
        self.detector.fit_benign(self.scaler.transform(vectors))
        return self

    def predict(self, summaries):
        vector = self.scaler.transform(aggregate(summaries, self.victim_keys, self.fields))
        score = float(self.detector.score(vector))
        return {'prediction': 1 if score >= 0 else -1, 'score': score, 'vector': vector}
