"""SiteGuard packet ingestion, feature extraction, TLS semantics and 2D-DFT.

This module does not decrypt traffic or infer missing handshake/certificate data.
The caller supplies parsed records observed by the current detection time.
"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
import pandas as pd


TLS_FIELDS = ('tls_has_heartbeat', 'tls_cipher_suites_cnt', 'tls_extensions_cnt',
              'tls_has_SNI', 'tls_sup_heartbeat', 'cert_self_signed',
              'cert_validity_duration', 'cert_SNI_in_SAN')


def extract_tls(metadata):
    """Read record types, ClientHello lists, and a decoded leaf certificate.

    Certificate times are Unix seconds. Issuer/subject must be canonical parsed
    distinguished names. trusted_root must be an explicit trust-store decision.
    SAN comparison uses case-insensitive exact DNS matching (no wildcard policy).
    """
    required = {'record_types', 'cipher_suites', 'extensions', 'server_name',
                'certificate'}
    if required - metadata.keys():
        raise ValueError('incomplete decoded TLS metadata')
    cert = metadata['certificate']
    if {'issuer', 'subject', 'trusted_root', 'not_before', 'not_after', 'dns_sans'} - cert.keys():
        raise ValueError('incomplete certificate metadata')
    if type(cert['trusted_root']) is not bool:
        raise ValueError('trusted_root must be an explicit boolean')
    lifetime = float(cert['not_after']) - float(cert['not_before'])
    if not math.isfinite(lifetime) or lifetime < 0:
        raise ValueError('invalid certificate validity interval')
    sni = metadata['server_name']
    extensions = metadata['extensions']
    values = [int(24 in metadata['record_types']), len(metadata['cipher_suites']),
              len(extensions), int(0 in extensions), int(15 in extensions),
              int(cert['issuer'] == cert['subject'] and not cert['trusted_root']),
              lifetime, int(bool(sni) and sni.lower().rstrip('.') in
                            {name.lower().rstrip('.') for name in cert['dns_sans']})]
    return dict(zip(TLS_FIELDS, values))


class SpectralTransform:
    def __init__(self, config):
        self.config = dict(config)
        self.minimum = self.maximum = None

    def magnitude(self, sequences):
        x = np.asarray(sequences, dtype=np.float64)
        if x.ndim != 3 or x.shape[1] != 5 or not np.isfinite(x).all():
            raise ValueError('expected finite (sessions, 5, K) sequences')
        z = np.abs(np.fft.fft2(x, axes=(-2, -1), norm=self.config['norm']))
        if self.config['frequency_order'] == 'paper_one_based':
            z = np.roll(z, (-1, -1), axis=(-2, -1))
        elif self.config['frequency_order'] != 'fft_zero_based':
            raise ValueError('unsupported frequency order')
        return z

    def fit(self, source_sequences):
        z = self.magnitude(source_sequences)
        if len(z) == 0:
            raise ValueError('no source sequences')
        self.minimum = z.min(axis=(0, 2))[None, :, None]
        self.maximum = z.max(axis=(0, 2))[None, :, None]
        return self

    def transform(self, sequences):
        if self.minimum is None:
            raise RuntimeError('fit source scaling before inference')
        z = self.magnitude(sequences)
        span = self.maximum - self.minimum
        scaled = (z - self.minimum) / np.where(span > 0, span, 1.0)
        if self.config['clip']:
            scaled = np.clip(scaled, 0, 1)
        lo, hi = self.config['c_min'], self.config['c_max']
        return (np.floor((hi - lo) * scaled) + lo).astype(np.float32)


PACKET_COLUMNS = ['timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port',
                  'protocol', 'packet_size', 'payload_size', 'tcp_flags', 'data_phase']


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(path, config):
    with Path(path).open(newline='') as stream:
        columns = next(csv.reader(stream))
    missing = sorted(set(PACKET_COLUMNS) - set(columns))
    return {'path': str(path), 'columns': columns, 'packet_input_supported': not missing,
            'missing_packet_columns': missing,
            'missing_tls_fields': sorted(set(config['features']['tls_fields']) - set(columns)),
            'multi_agent_key_available': 'agent_id' in columns,
            'limitations': [
                'Flow totals, mean sizes and mean IATs cannot reconstruct the first K packets.',
                'Flow timestamps do not establish per-packet detection-window counts.',
                'TLS certificate and ClientHello semantics require decoded handshake metadata.',
                'No automatic semantic mapping from flow columns to Table 4 is performed.'] if missing else []}


def read_packets(path, config):
    report = audit(path, config)
    if not report['packet_input_supported']:
        raise ValueError('Packet-level CSV required; missing: ' + ', '.join(report['missing_packet_columns']))
    frame = pd.read_csv(path, dtype={'src_ip': str, 'dst_ip': str})
    if frame.empty or frame[PACKET_COLUMNS].isna().any().any():
        raise ValueError('empty input or missing required packet values')
    numeric = ['timestamp', 'src_port', 'dst_port', 'protocol', 'packet_size',
               'payload_size', 'tcp_flags', 'data_phase']
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError('packet metadata must be finite')
    if (frame[numeric] < 0).any().any() or not frame.timestamp.is_monotonic_increasing:
        raise ValueError('packet metadata must be nonnegative and chronologically ordered')
    integers = numeric[1:]
    if (frame[integers] % 1 != 0).any().any():
        raise ValueError('ports, protocol, lengths, flags and data_phase must be integers')
    if (frame[['src_port', 'dst_port']] > 65535).any().any() or (frame.protocol > 255).any():
        raise ValueError('invalid port/protocol number')
    if (frame.payload_size > frame.packet_size).any() or not frame.data_phase.isin([0, 1]).all():
        raise ValueError('invalid payload length or data_phase flag')
    if 'label' in frame and not frame.label.isin([-1, 1]).all():
        raise ValueError('packet labels must explicitly use -1 benign / +1 anomalous')
    if 'agent_id' in frame and (frame.agent_id.isna().any() or frame.agent_id.nunique() != 1):
        raise ValueError('local input must contain exactly one agent; aggregate agents separately')
    if config['features']['tls_enabled']:
        fields = config['features']['tls_fields']
        if 'tls_metadata' in frame:
            decoded = pd.DataFrame([extract_tls(json.loads(value)) for value in frame.tls_metadata])
            for name in fields:
                frame[name] = decoded[name].to_numpy()
        if set(fields) - set(frame):
            raise ValueError('TLS enabled but decoded session metadata is missing')
        if not np.isfinite(frame[fields].to_numpy(dtype=float)).all():
            raise ValueError('TLS metadata must be complete and finite; unknown is not zero')
    return frame


@dataclass
class DetectionWindow:
    window_id: str
    start: float
    end: float
    statistics: np.ndarray
    sequences: np.ndarray
    tls: np.ndarray
    digest: dict
    label: int | None
    session_count: int


def aggregated_counts(frame, config):
    """Table 4 keyed counters, reduced by the declared window aggregation policy."""
    if frame.empty:
        return np.zeros(len(config['granularities']) * len(config['measurements'])), {}
    keys = {
        'dst_ip': ['dst_ip'], 'dst_port': ['dst_port'],
        'src_dst_ip': ['src_ip', 'dst_ip'], 'src_sock': ['src_ip', 'src_port'],
        'dst_sock': ['dst_ip', 'dst_port'],
    }
    values = []
    for granularity in config['granularities']:
        port = 'src_port' if granularity in ('dst_port', 'dst_sock') else 'dst_port'
        grouped = frame.groupby(keys[granularity], sort=False)
        counts = grouped.agg(pkts=('packet_size', 'size'), bytes=('packet_size', 'sum'),
                             ports=(port, 'nunique'), src_ips=('src_ip', 'nunique'))
        if granularity == 'src_dst_ip':
            # Union source and destination ports within each endpoint pair.
            ports = pd.concat([
                frame[['src_ip', 'dst_ip', port]].rename(columns={port: 'port'})
                for port in ('src_port', 'dst_port')], ignore_index=True)
            counts['ports'] = ports.groupby(keys[granularity], sort=False)['port'].nunique()
        a = counts.to_numpy(dtype=float)
        for i, reduction in enumerate(config['group_reduction']):
            values.append(float(getattr(a[:, i], reduction)()) if len(a) else 0.0)
    digest = frame.groupby('dst_ip', sort=True).agg(
        packets=('packet_size', 'size'), bytes=('packet_size', 'sum')).to_dict('index')
    return np.asarray(values), digest


class PacketWindows:
    """Causal historical rates and bidirectional sessions with idle/FIN/RST expiry.

    Captures first K data-phase packets per session per window; history never uses
    future windows. Interval state survives boundaries; short records are padded.
    """
    def __init__(self, config):
        self.config = config

    def extract(self, frame):
        f = self.config['features']
        duration, k = f['window_seconds'], f['sequence_length']
        rng = np.random.default_rng(self.config['seed'])
        history = deque(maxlen=f['history_windows'])
        sessions = {}
        generation = 0
        bin_ids = np.floor(frame.timestamp.to_numpy() / duration).astype(np.int64)
        # Include genuinely empty intervening windows; omit unavailable edge fragments.
        first = int(np.ceil(float(frame.timestamp.iloc[0]) / duration))
        last = int(np.floor(float(frame.timestamp.iloc[-1]) / duration))
        # Keep row boundaries rather than a copied DataFrame for every window.
        starts = np.r_[0, np.flatnonzero(np.diff(bin_ids)) + 1]
        stops = np.r_[starts[1:], len(frame)]
        bounds = {int(bin_ids[start]): (start, stop) for start, stop in zip(starts, stops)}
        windows = []
        for index in range(first, last):
            start, stop = bounds.get(index, (0, 0))
            group = frame.iloc[start:stop]
            counts, digest = aggregated_counts(group, f)
            rate = counts / duration
            baseline = np.mean(history, axis=0) if history else np.zeros(20)
            deviation = np.abs(rate - baseline) / np.maximum(baseline, f['zero_baseline_epsilon'])
            stats = np.stack([counts, rate, baseline, deviation], axis=1).reshape(-1)
            ready = len(history) == f['history_windows']
            history.append(rate)
            # Expire idle sessions even when a window has no traffic.
            sessions = {key: state for key, state in sessions.items()
                        if index * duration - state['last'] < f['session_timeout_seconds']}
            records, semantic = {}, {}
            for row in group.to_dict('records'):
                endpoints = tuple(sorted(((row['src_ip'], int(row['src_port'])),
                                          (row['dst_ip'], int(row['dst_port'])))))
                key = (endpoints, int(row['protocol']))
                state = sessions.get(key)
                t = float(row['timestamp'])
                if state is None or t - state['last'] >= f['session_timeout_seconds']:
                    generation += 1
                    state = {'last': t, 'data_last': None, 'interval': 0.0, 'generation': generation}
                    sessions[key] = state
                record_key = (key, state['generation'])
                state['last'] = t
                if row['data_phase']:
                    interval = 0.0 if state['data_last'] is None else t - state['data_last']
                    jitter = 0.0 if state['data_last'] is None else interval - state['interval']
                    record = records.setdefault(record_key, [])
                    if len(record) < k:
                        record.append([row['packet_size'], interval, jitter, row['protocol'], row['payload_size']])
                    state['data_last'], state['interval'] = t, interval
                    if f['tls_enabled']:
                        semantic[record_key] = [row[name] for name in f['tls_fields']]
                if int(row['protocol']) == 6 and int(row['tcp_flags']) & 5:
                    sessions.pop(key, None)
            if not ready:
                continue  # explicit causal warmup, no fictitious historical rates
            record_keys = list(records)
            n = min(len(record_keys), f['sampled_sessions'])
            # Seeded uniform sampling without replacement, then arrival order.
            selected = sorted(rng.choice(len(record_keys), n, replace=False).tolist())
            sequences = np.zeros((n, 5, k), np.float32)
            tls = np.zeros((n, 8 if f['tls_enabled'] else 0), np.float32)
            for out, selected_index in enumerate(selected):
                key = record_keys[selected_index]
                values = np.asarray(records[key], dtype=np.float32).T
                sequences[out, :, :values.shape[1]] = values
                if f['tls_enabled']:
                    tls[out] = semantic[key]
            label = (1 if (group.label == 1).any() else -1) if 'label' in frame else None
            windows.append(DetectionWindow(str(index), index * duration, (index + 1) * duration,
                stats, sequences, tls, digest, label, len(records)))
        if not windows:
            raise ValueError('no complete detection windows after causal history warmup')
        return windows
