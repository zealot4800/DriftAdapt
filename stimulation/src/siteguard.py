"""SiteGuard pipeline integrated with the shared simulation runner/configuration."""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import torch
from .siteguard_models import SequenceAutoencoder, OneClassDetector, SourceScaler
from .siteguard_features import PacketWindows, fingerprint, read_packets, SpectralTransform, TLS_FIELDS
from .siteguard_update import OnlineUpdater


def siteguard_settings(shared_config):
    """Extract model settings from the shared simulation dataset configuration."""
    section = shared_config.get('siteguard')
    required = ('features', 'dft', 'autoencoder', 'classifier', 'online', 'server')
    if not isinstance(section, dict) or any(not isinstance(section.get(key), dict) for key in required):
        raise ValueError('siteguard requires features, dft, autoencoder, classifier, online and server sections')
    config = {key: copy.deepcopy(section[key]) for key in
              required}
    config['seed'] = int(shared_config.get('runtime', {}).get('seed', 42))
    config['threads'] = int(section.get('threads', 1))
    f = config['features']
    for key in ('window_seconds', 'sequence_length', 'sampled_sessions',
                'session_timeout_seconds', 'history_windows', 'zero_baseline_epsilon'):
        if f[key] <= 0:
            raise ValueError(f'features.{key} must be positive')
    if f['granularities'] != ['dst_ip', 'dst_port', 'src_dst_ip', 'src_sock', 'dst_sock']:
        raise ValueError('granularities must retain Table 4 ordering')
    if f['measurements'] != ['pkts', 'bytes', 'ports', 'src_ips']:
        raise ValueError('measurements must retain Table 4 ordering')
    if f['sequence_order'] != ['packet_size', 'interval_time', 'jitter', 'protocol', 'payload_size']:
        raise ValueError('sequence_order must retain Table 4 ordering')
    if len(f['group_reduction']) != 4 or set(f['group_reduction']) - {'sum', 'max'}:
        raise ValueError('group_reduction needs four sum/max operations')
    if f['short_window_policy'] != 'zero_pad_embeddings' or f['window_label_policy'] != 'any_attack':
        raise ValueError('unsupported window policy')
    if len(f['tls_fields']) != 8:
        raise ValueError('exactly eight ordered TLS fields required')
    if tuple(f['tls_fields']) != TLS_FIELDS:
        raise ValueError('TLS fields must retain Table 4 ordering')
    if config['classifier']['objective'] != 'slack_corrected_linear':
        raise ValueError('only the documented slack-corrected objective is implemented')
    if not 0 < config['classifier']['nu'] <= 1:
        raise ValueError('nu must be in (0, 1]')
    a = config['autoencoder']
    if any(a[k] <= 0 for k in ('channels', 'latent_dim', 'epochs', 'batch_size', 'learning_rate')) or a['residual_blocks'] < 0:
        raise ValueError('invalid autoencoder parameters')
    o = config['online']
    if not (0 <= o['beta'] <= 1 and 0 <= o['delta'] <= 1):
        raise ValueError('beta and delta must be probabilities')
    if o['schedule'] not in ('constant', 'inverse_sqrt') or o['learning_rate'] <= 0:
        raise ValueError('invalid learning rate schedule')
    if not 1 <= o['minimum_benign_feedback'] <= o['feedback_window']:
        raise ValueError('invalid feedback window/minimum')
    if config['dft']['c_max'] <= config['dft']['c_min']:
        raise ValueError('invalid DFT scaling range')
    return config


class SiteGuard:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        torch.manual_seed(config['seed'])
        torch.set_num_threads(config['threads'])
        torch.use_deterministic_algorithms(True)
        self.spectral = SpectralTransform(config['dft'])
        self.autoencoder = SequenceAutoencoder(config['autoencoder'], config['features']['sequence_length'])
        self.scaler = SourceScaler()
        self.detector = OneClassDetector(config['classifier'])
        self.metadata = {}

    def spectra(self, sequences):
        z = self.spectral.transform(sequences)
        # AE numerical conditioning follows quantization; separately documented.
        return (z - self.config['dft']['c_min']) / (self.config['dft']['c_max'] - self.config['dft']['c_min'])

    def raw_vector(self, window):
        f = self.config['features']
        latent = self.autoencoder.encode(self.spectra(window.sequences))
        per_session = np.concatenate([latent, window.tls], axis=1)
        padded = np.zeros((f['sampled_sessions'], per_session.shape[1]), dtype=np.float64)
        padded[:len(per_session)] = per_session
        return np.concatenate([padded.reshape(-1), window.statistics])

    def fit(self, source_path, *, verified_benign):
        if not verified_benign:
            raise ValueError('training requires an explicitly verified benign source partition')
        frame = read_packets(source_path, self.config)
        if 'label' in frame and (frame.label != -1).any():
            raise ValueError('source contains attack labels; provide a separate benign source partition')
        windows = PacketWindows(self.config).extract(frame)
        sequences = np.concatenate([w.sequences for w in windows])
        self.spectral.fit(sequences)
        loss = self.autoencoder.fit_source(self.spectra(sequences), self.config['seed'])
        vectors = np.stack([self.raw_vector(w) for w in windows])
        self.scaler.fit(vectors)
        self.detector.fit_benign(self.scaler.transform(vectors))
        self.metadata = {'format_version': 1, 'source_sha256': fingerprint(source_path),
            'source_start': float(frame.timestamp.iloc[0]), 'source_end': float(frame.timestamp.iloc[-1]),
            'training_windows': len(windows), 'source_verified_benign': True,
            'reconstruction_mse': loss,
            'fidelity': 'documented software reproduction with corrected one-class objective',
            'tls_enabled': self.config['features']['tls_enabled']}
        return self.metadata

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(f'refusing to overwrite checkpoint: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'config': self.config, 'metadata': self.metadata,
                    'autoencoder': self.autoencoder.state_dict(),
                    'dft_min': torch.from_numpy(self.spectral.minimum),
                    'dft_max': torch.from_numpy(self.spectral.maximum),
                    'scaler_min': torch.from_numpy(self.scaler.minimum),
                    'scaler_span': torch.from_numpy(self.scaler.span),
                    'weights': torch.from_numpy(self.detector.weights), 'rho': self.detector.rho}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['metadata']['format_version'] != 1:
            raise ValueError('unsupported checkpoint format')
        model = cls(saved['config'])
        model.metadata = saved['metadata']
        model.autoencoder.load_state_dict(saved['autoencoder'])
        model.autoencoder.eval()
        model.spectral.minimum = saved['dft_min'].numpy()
        model.spectral.maximum = saved['dft_max'].numpy()
        model.scaler.minimum = saved['scaler_min'].numpy()
        model.scaler.span = saved['scaler_span'].numpy()
        model.detector.weights = saved['weights'].numpy().copy()
        model.detector.rho = saved['rho']
        return model


def read_feedback(path, windows):
    known = {w.window_id: w for w in windows}
    records, seen = [], set()
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            key = str(record['window_id'])
            if key not in known or key in seen:
                raise ValueError('feedback references an unknown or duplicate detection window')
            if record['verified'] is not True or type(record['label']) is not int or record['label'] not in (-1, 1):
                raise ValueError('feedback must explicitly be verified and use -1/+1 labels')
            available = float(record['available_at'])
            if not np.isfinite(available) or available < known[key].end:
                raise ValueError('feedback predates the detection window or has invalid availability')
            record.update(window_id=key, available_at=available)
            records.append(record)
            seen.add(key)
    return sorted(records, key=lambda r: r['available_at'])


def evaluate(model, packet_path, output_dir, feedback_path=None):
    """Replay once and return window records plus their persisted summary."""
    frame = read_packets(packet_path, model.config)
    evaluation_fingerprint = fingerprint(packet_path)
    if evaluation_fingerprint == model.metadata['source_sha256']:
        raise ValueError('evaluation file duplicates the training source')
    if float(frame.timestamp.iloc[0]) <= model.metadata.get('last_observed_end', model.metadata['source_end']):
        raise ValueError('evaluation must be a disjoint chronological partition after source training')
    windows = PacketWindows(model.config).extract(frame)
    feedback = read_feedback(feedback_path, windows) if feedback_path else []
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    updater = OnlineUpdater(model.detector, model.config['online'], destination / 'updates') if feedback_path else None
    predictions, pending = [], {}
    cursor = 0

    def apply_available(now):
        nonlocal cursor
        while cursor < len(feedback) and feedback[cursor]['available_at'] <= now:
            record = feedback[cursor]
            if record['window_id'] not in pending:
                raise ValueError('feedback became available before its prediction was produced')
            vector, prediction, end = pending.pop(record['window_id'])
            updater.observe(window_id=record['window_id'], vector=vector, prediction=prediction,
                window_end=end, available_at=record['available_at'], now=now,
                label=record['label'], verified=record['verified'])
            cursor += 1

    feedback_ids = {r['window_id'] for r in feedback}
    with (destination / 'predictions.jsonl').open('w') as stream:
        for window in windows:
            # Feedback available at the start affects this window; feedback arriving
            # during it is processed only after its prediction (conservative causality).
            apply_available(window.start)
            vector = model.scaler.transform(model.raw_vector(window))
            score = float(model.detector.score(vector))
            prediction = 1 if score >= 0 else -1
            row = {'window_id': window.window_id, 'start': window.start, 'end': window.end,
                   'prediction': prediction, 'score': score, 'label': window.label,
                   'session_count': window.session_count, 'sampled_sessions': len(window.sequences),
                   'model_version': updater.version if updater else 0,
                   'unit': 'detection_window'}
            predictions.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            if window.window_id in feedback_ids:
                pending[window.window_id] = (vector, prediction, window.end)
            apply_available(window.end)
    labeled = [r for r in predictions if r['label'] is not None]
    metrics = None
    if labeled:
        tp = sum(r['prediction'] == 1 and r['label'] == 1 for r in labeled)
        fp = sum(r['prediction'] == 1 and r['label'] == -1 for r in labeled)
        fn = sum(r['prediction'] == -1 and r['label'] == 1 for r in labeled)
        tn = len(labeled) - tp - fp - fn
        metrics = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                   'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                   'false_positive_rate': fp / (fp + tn) if fp + tn else None}
    summary = {'method': 'siteguard', 'dataset': model.metadata.get('dataset'),
               'action': 'adapt' if updater else 'infer',
               'mode': 'adaptive' if updater else 'frozen', 'unit': 'detection_window',
               'windows': len(windows), 'metrics': metrics,
               'feedback_applied': cursor, 'feedback_not_yet_available': len(feedback) - cursor,
               'updates': updater.version if updater else 0,
               'evaluation_sha256': evaluation_fingerprint, 'training': model.metadata}
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if updater:
        model.metadata = dict(model.metadata, online_summary=summary, last_observed_end=windows[-1].end)
        model.save(destination / 'adapted.pt')
    return pd.DataFrame.from_records(predictions), summary


class SiteGuardPipeline:
    """SiteGuard adapter for the existing run.py/config/pipeline interface.

    Uses shared dataset names, runtime seeds and output root. Packet-specific
    configuration stays under siteguard; flow loaders and student checkpoints
    remain exclusive to CARAVAN/DriftAdapt.
    """
    def __init__(self, config, config_path):
        self.config = config
        self.config_path = Path(config_path).resolve()
        self.settings = siteguard_settings(config)
        self.section = config['siteguard']

    def path(self, value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.config_path.parent / path).resolve()

    def run(self):
        from .siteguard_features import audit
        action = self.section.get('action', 'infer')
        if action not in {'audit', 'train', 'infer', 'adapt'}:
            raise ValueError('siteguard.action must be audit, train, infer, or adapt')
        dataset = self.config['dataset']['name']
        if action == 'audit':
            configured = self.section.get('evaluation_packets')
            paths = [configured] if configured else self.config['dataset']['files']
            reports = [audit(self.path(path), self.settings) for path in paths]
            return reports, {'method': 'siteguard', 'dataset': dataset, 'action': action,
                             'inputs': reports}
        source_key = 'source_packets' if action == 'train' else 'evaluation_packets'
        configured = self.section.get(source_key)
        if not configured:
            raise ValueError(
                f'SiteGuard requires siteguard.{source_key} or --packets with packet-level CSV input. '
                'The bundled dataset.files contain flow summaries, not packet sequences. '
                'Use --mode siteguard --action audit to inspect them.')
        packets = self.path(configured)
        # Check inputs before loading/training a model, so missing packet metadata
        # is explained even if the user has not created a SiteGuard checkpoint yet.
        report = audit(packets, self.settings)
        if not report['packet_input_supported']:
            raise ValueError('SiteGuard packet CSV is missing: ' + ', '.join(report['missing_packet_columns']))
        checkpoint = self.section.get('checkpoint')
        if not checkpoint:
            raise ValueError('siteguard.checkpoint or --checkpoint is required')
        checkpoint = self.path(checkpoint)
        if action == 'train':
            if checkpoint.exists():
                raise FileExistsError(f'refusing to overwrite checkpoint: {checkpoint}')
            model = SiteGuard(self.settings)
            summary = model.fit(packets, verified_benign=self.section.get('verified_benign_source') is True)
            model.metadata['dataset'] = dataset
            model.save(checkpoint)
            return [], dict(summary, method='siteguard', dataset=dataset,
                            action=action, checkpoint=str(checkpoint))
        feedback = self.section.get('feedback')
        if action == 'adapt' and not feedback:
            raise ValueError('SiteGuard adaptation requires siteguard.feedback or --feedback')
        model = SiteGuard.load(checkpoint)
        if model.metadata.get('dataset', dataset) != dataset:
            raise ValueError('SiteGuard checkpoint belongs to a different dataset')
        model.metadata.setdefault('dataset', dataset)
        # Checkpoint preprocessing/model configuration is authoritative at inference.
        # Shared YAML settings remain the source of configuration when training.
        if self.section.get('output'):
            output = self.path(self.section['output'])
        else:
            output = Path(self.config.get('output', {}).get('directory', 'results'))
            if not output.is_absolute():
                output = self.config_path.parent.parent / output
            output = output / f'{dataset}-siteguard-{action}'
        return evaluate(model, packets, output,
                        self.path(feedback) if action == 'adapt' else None)
