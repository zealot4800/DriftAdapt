"""Shared packet ingestion, chronological windows and source-fitted encoding."""
from collections import deque
import math
import numpy as np
import pandas as pd
import torch
from .config import path_for
from .models import PacketEncoder

TLS_FIELDS = (
    'tls_has_heartbeat', 'tls_cipher_suites_cnt', 'tls_extensions_cnt',
    'tls_has_SNI', 'tls_sup_heartbeat', 'cert_self_signed',
    'cert_validity_duration', 'cert_SNI_in_SAN',
)
PACKET_FIELDS = (
    'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
    'packet_size', 'payload_size', 'tcp_flags', 'data_phase',
)


class PacketFeatures:
    def __init__(self, settings, seed):
        self.seconds = settings['window_seconds']
        self.length = settings['sequence_length']
        self.sample_count = settings['sampled_sessions']
        self.timeout = settings['session_timeout_seconds']
        if (not np.isfinite([self.seconds, self.timeout]).all()
                or self.seconds <= 0 or self.timeout <= 0
                or any(type(settings[key]) is not int or settings[key] < 1
                       for key in ('sequence_length', 'sampled_sessions', 'history_windows'))):
            raise ValueError('Packet window, sequence, sampling, timeout and history settings must be positive')
        self.history = deque(maxlen=settings['history_windows'])
        self.sessions = {}
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def counts(frame):
        values = []
        packet_count, byte_count = len(frame), frame.packet_size.sum()
        source_present = frame.src_ip.notna().all()
        for columns in (['dst_ip'], ['dst_port'], ['src_ip', 'dst_ip'],
                        ['src_ip', 'src_port'], ['dst_ip', 'dst_port']):
            groups = frame.groupby(columns, sort=False)
            if columns == ['src_ip', 'dst_ip']:
                ports = max((len(set(group.src_port) | set(group.dst_port))
                             for _, group in groups), default=0)
            else:
                port = 'dst_port' if columns in (['dst_ip'], ['src_ip', 'src_port']) else 'src_port'
                ports = groups[port].nunique().max() if packet_count else 0
            sources = (1 if source_present and 'src_ip' in columns else
                       groups.src_ip.nunique().max()) if packet_count else 0
            values.extend((packet_count, byte_count, ports, sources))
        return np.asarray(values, dtype=np.float64)

    def extract(self, frame, start):
        self.sessions = {key: value for key, value in self.sessions.items()
                         if start - value['last'] <= self.timeout}
        current = {}
        for row in frame.itertuples(index=False):
            sockets = sorted(((row.src_ip, row.src_port), (row.dst_ip, row.dst_port)))
            key = (*sockets, row.protocol)
            state = self.sessions.get(key)
            if state is None or row.timestamp - state['last'] > self.timeout:
                state = {'last': row.timestamp, 'rows': [], 'time': None,
                         'interval': None, 'tls': np.zeros(len(TLS_FIELDS))}
                self.sessions[key] = state
            state['last'] = row.timestamp
            state['tls'] = np.maximum(state['tls'], [getattr(row, field) for field in TLS_FIELDS])
            if row.data_phase and len(state['rows']) < self.length:
                interval = row.timestamp - state['time'] if state['time'] is not None else 0.0
                jitter = abs(interval - state['interval']) if state['interval'] is not None else 0.0
                state['rows'].append((row.packet_size, interval, jitter,
                                      row.protocol, row.payload_size))
                state['interval'] = interval if state['time'] is not None else None
                state['time'] = row.timestamp
            current[id(state)] = state
            if row.protocol == 6 and int(row.tcp_flags) & 5:
                self.sessions.pop(key, None)
        flows = [state for state in current.values() if state['rows']]
        selected = sorted(self.rng.choice(len(flows), min(len(flows), self.sample_count), replace=False))
        sequences = np.zeros((self.sample_count, 5, self.length), dtype=np.float32)
        mask = np.zeros(self.sample_count, dtype=bool)
        for slot, index in enumerate(selected):
            rows = np.asarray(flows[index]['rows'], dtype=np.float32).T
            sequences[slot, :, :rows.shape[1]] = rows
            mask[slot] = True
        tls = np.max([state['tls'] for state in current.values()], axis=0) if current else np.zeros(len(TLS_FIELDS))
        counts = self.counts(frame)
        rate = counts / self.seconds
        baseline = np.mean(self.history, axis=0) if self.history else np.zeros(20)
        deviation = np.abs(rate - baseline) / np.maximum(baseline, 1.0 / self.seconds)
        self.history.append(rate)
        statistics = np.stack((counts, rate, baseline, deviation), axis=1).reshape(-1)
        return sequences, mask, np.concatenate((statistics, tls)).astype(np.float32)


class PacketData:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['runtime']['device'])
        self.timeline = config['timeline']
        packet_file = config['dataset'].get('packet_file')
        if not packet_file:
            raise ValueError('Set dataset.packet_file to a chronological packet metadata CSV; flow CSVs are insufficient')
        if (self.timeline.get('order') != 'timestamp' or
                any(self.timeline.get(key) is None for key in ('source_end', 'adapter_end'))):
            raise ValueError('Set timeline.source_end and adapter_end to packet timestamp boundaries in seconds')
        label = config['dataset'].get('label_column', 'label')
        frame = pd.read_csv(path_for(config, packet_file), low_memory=False)
        frame.columns = frame.columns.str.strip()
        required = (*PACKET_FIELDS, *TLS_FIELDS, label)
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing packet columns: {", ".join(sorted(missing))}')
        frame = frame[list(required)].copy()
        if frame.empty or frame.drop(columns=label).isna().any().any():
            raise ValueError('Packet metadata must be nonempty without missing features; use explicit zero TLS values for non-TLS traffic')
        numeric = [name for name in required if name not in ('src_ip', 'dst_ip')]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors='raise')
        if (not np.isfinite(frame[numeric].drop(columns=label).to_numpy()).all()
                or not frame.timestamp.is_monotonic_increasing):
            raise ValueError('Packet features must be finite with chronological numeric timestamps in seconds')
        integers = ('src_port', 'dst_port', 'protocol', 'tcp_flags', 'packet_size',
                    'payload_size', 'data_phase', *[field for field in TLS_FIELDS if field != 'cert_validity_duration'])
        if any((frame[name] % 1 != 0).any() for name in integers):
            raise ValueError('Packet ports, protocol, flags, lengths, phase and TLS counts must be integers')
        if ((frame[numeric].drop(columns=['timestamp', label]) < 0).any().any()
                or any(not frame[field].isin([0, 1]).all() for field in
                       ('data_phase', 'tls_has_heartbeat', 'tls_has_SNI', 'tls_sup_heartbeat',
                        'cert_self_signed', 'cert_SNI_in_SAN'))
                or (frame.payload_size > frame.packet_size).any()
                or (frame[['src_port', 'dst_port']] > 65535).any().any()
                or (frame[['protocol', 'tcp_flags']] > 255).any().any()
                or not frame[label].dropna().isin([0, 1]).all()):
            raise ValueError('Invalid packet lengths, ports, protocol, flags, TLS metadata or binary labels (benign=0, attack=1)')
        frame['src_ip'] = frame.src_ip.astype(str)
        frame['dst_ip'] = frame.dst_ip.astype(str)
        extractor = PacketFeatures(config['features'], config['runtime']['seed'])
        seconds = extractor.seconds
        end = self.timeline.get('evaluation_end')
        if end is None:
            end = math.floor(frame.timestamp.iloc[-1] / seconds) * seconds
        bounds = [math.ceil(frame.timestamp.iloc[0] / seconds) * seconds,
                  self.timeline['source_end'], self.timeline['adapter_end'], end]
        if (any(not math.isfinite(value) for value in bounds)
                or any(right <= left for left, right in zip(bounds, bounds[1:]))
                or any(not math.isclose(value / seconds, round(value / seconds), abs_tol=1e-7, rel_tol=0) for value in bounds)):
            raise ValueError('Packet timeline boundaries must increase and align with complete detection windows')
        if end > frame.timestamp.iloc[-1] + seconds:
            raise ValueError('Packet evaluation_end extends beyond the available capture')
        self.frames = tuple(frame[(frame.timestamp >= left) & (frame.timestamp < right)].reset_index(drop=True)
                            for left, right in zip(bounds, bounds[1:]))
        if any(part.empty for part in self.frames):
            raise ValueError('Packet source, offline and test partitions must be nonempty')
        for part in self.frames[:2]:
            if part[label].isna().any():
                raise ValueError('Source and offline packet labels are required; only test labels may be missing')
            part[label] = part[label].astype(np.int64)
        self.partitions = []
        for part, left, right in zip(self.frames, bounds, bounds[1:]):
            windows = []
            timestamps = part.timestamp.to_numpy()
            for index in range(round((right - left) / seconds)):
                start = left + index * seconds
                begin, stop = timestamps.searchsorted([start, start + seconds], side='left')
                window = part.iloc[begin:stop]
                sequences, mask, statistics = extractor.extract(window, start)
                target = 1 if (window[label] == 1).any() else None if window[label].isna().any() else 0
                windows.append({'start': start, 'end': start + seconds,
                                'sequences': sequences, 'mask': mask,
                                'statistics': statistics, 'label': target})
            self.partitions.append(windows)
        self.labels = [torch.tensor([w['label'] if w['label'] is not None else -1 for w in part],
                                   dtype=torch.long, device=self.device) for part in self.partitions]
        self.dropped_partial_packets = len(frame) - sum(map(len, self.frames))
        self.ready = False

    def prepare(self):
        if self.ready:
            return
        settings = self.config['features']
        encoder = PacketEncoder(settings, self.device)
        encoder.fit(self.partitions[0])
        self.encoder = encoder
        encoder.eval()
        vectors = [np.stack([encoder.vector(w) for w in part]) for part in self.partitions]
        if not all(np.isfinite(part).all() for part in vectors):
            raise ValueError('Nonfinite shared packet features')
        mean, std = vectors[0].mean(axis=0), vectors[0].std(axis=0)
        std = np.where(std > 0, std, 1.)
        self.scale = torch.from_numpy(std).to(self.device)
        self.raw = [torch.from_numpy(part).to(self.device) for part in vectors]
        self.features = [torch.from_numpy((part-mean)/std).to(self.device) for part in vectors]
        self.feature_names = [f'{group}_{value}_{stat}'
                              for group in ('destination_ip', 'destination_port', 'ip_pair', 'source_socket', 'destination_socket')
                              for value in ('packet_count', 'byte_count', 'port_count', 'source_count')
                              for stat in ('count', 'rate', 'history', 'deviation')]
        self.feature_names += list(TLS_FIELDS)
        self.feature_names += [f'session_{slot}_embedding_{coordinate}'
                               for slot in range(settings['sampled_sessions'])
                               for coordinate in range(settings['autoencoder']['latent_dim'])]
        self.ready = True

    def observations(self):
        if not self.ready:
            raise RuntimeError('Prepare the shared packet encoder before inference')
        for index, window in enumerate(self.partitions[2]):
            yield {'id': index+1, 'start': window['start'], 'end': window['end'],
                   'features': self.features[2][index:index+1],
                   'raw_features': self.raw[2][index:index+1]}, [window['label']]
