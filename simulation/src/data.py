"""Chronological partitions and preprocessing fitted exclusively on source data."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd
import torch
from .config import path_for

_PACKET_COLUMNS = ('timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'packet_size', 'payload_size', 'tcp_flags', 'data_phase')


@dataclass
class Window:
    features: torch.Tensor
    labels: torch.Tensor
    raw_features: torch.Tensor
    generated_labels: torch.Tensor


def frame_fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def partition_frame(frame, timeline, timestamp=None):
    if timestamp:
        values = pd.to_numeric(frame[timestamp], errors='raise')
        if not np.isfinite(values).all() or not values.is_monotonic_increasing:
            raise ValueError('Timestamps must be finite and chronological; input is never shuffled')
        source, adapter = timeline['source_end'], timeline['adapter_end']
        end = timeline.get('evaluation_end')
        if not source < adapter or (end is not None and end <= adapter):
            raise ValueError('Timeline boundaries must be strictly increasing')
        parts = (frame[values < source], frame[(values >= source) & (values < adapter)],
                 frame[(values >= adapter) & (True if end is None else values < end)])
    else:
        if timeline['order'] != 'provided':
            raise ValueError('Timestamp chronology requires a timestamp column')
        source, adapter = int(timeline['source_rows']), int(timeline['adapter_rows'])
        if source < 2 or adapter < 1:
            raise ValueError('Source and adapter partitions must be nonempty')
        parts = (frame.iloc[:source], frame.iloc[source:source + adapter], frame.iloc[source + adapter:])
    if any(part.empty for part in parts):
        raise ValueError('Source, offline-adapter, and evaluation partitions must be disjoint and nonempty')
    return tuple(part.reset_index(drop=True) for part in parts)


class FlowData:
    def __init__(self, config):
        section = config['dataset']
        self.device = torch.device(config['runtime']['device'])
        self.feature_names = list(section['feature_names'])
        names = [name.strip().lower() for name in self.feature_names]
        if section.get('packet_file'):
            raise ValueError('The simulation supports flow CSV datasets only')
        frames = [pd.read_csv(path_for(config, value)) for value in section['files']]
        frame = pd.concat(frames, ignore_index=True)
        frame.columns = frame.columns.str.strip().str.lower()
        if 'total packet length' in names and 'total packet length' not in frame:
            frame['total packet length'] = frame['total length of fwd packets'] + frame['total length of bwd packets']
        if 'number of packets' in names and 'number of packets' not in frame:
            frame['number of packets'] = frame['total fwd packets'] + frame['total backward packets']
        timestamp = section.get('timestamp_column')
        parts = partition_frame(frame, config['timeline'], timestamp)
        self.timestamp_column = timestamp
        self.frames = parts
        columns = list(parts[0].columns)
        self.schema_report = {'columns':columns, 'packet_columns_present':not (set(_PACKET_COLUMNS)-set(columns)),
                              'missing_packet_columns':sorted(set(_PACKET_COLUMNS)-set(columns)),
                              'representation':'flow', 'packet_sequences_reconstructed':False}
        raw = [torch.tensor(p[names].to_numpy(dtype='float32')) for p in parts]
        if any(not torch.isfinite(x).all() for x in raw):
            raise ValueError('Flow features must be finite')
        source = raw[0]
        self.mean = source.mean(0, keepdim=True) if section['standardize'] else torch.zeros(1, len(names))
        self.std = source.std(0, keepdim=True) if section['standardize'] else torch.ones(1, len(names))
        self.std = torch.where(self.std > 0, self.std, torch.ones_like(self.std))
        transformed = [(x-self.mean)/self.std for x in raw]
        self.minimum = transformed[0].amin(0, keepdim=True) if section['normalize'] else torch.zeros_like(self.mean)
        self.span = transformed[0].amax(0, keepdim=True)-self.minimum if section['normalize'] else torch.ones_like(self.mean)
        self.span = torch.where(self.span > 0, self.span, torch.ones_like(self.span))
        self.features = [((x-self.minimum)/self.span).to(self.device) for x in transformed]
        self.raw = [x.to(self.device) for x in raw]
        label = section.get('label_column', 'label').lower()
        self.labels = []
        self.generated = []
        for part in parts:
            if label not in part:
                raise ValueError(f'Flow evaluation/training requires the configured label column: {label}')
            values = part[label]
            if section.get('class_mapping'):
                values = values.map(section['class_mapping'])
            if values.isna().any() or not values.isin(set(range(section['num_classes'])) | set(benign_classes(config) or [])).all():
                raise ValueError('Labels must match the explicitly configured class mapping')
            self.labels.append(torch.tensor(values.to_numpy(), dtype=torch.long, device=self.device))
            column = section.get('generated_label_column')
            self.generated.append(torch.tensor(part[column].to_numpy(), dtype=torch.long, device=self.device)
                                  if column else torch.full_like(self.labels[-1], -1))
        if any(not ((labels >= -1) & (labels < section['num_classes'])).all() for labels in self.generated):
            raise ValueError('Generated teacher labels must be -1 or a configured class index')
        metadata = {'protocol': 'source_only_v1', 'feature_names': self.feature_names,
                    'class_mapping': section.get('class_mapping'), 'num_classes': section['num_classes'],
                    'source_sha256': frame_fingerprint(parts[0]),
                    'statistics': [x.tolist() for x in (self.mean,self.std,self.minimum,self.span)]}
        self.preprocessing = metadata
        self.preprocessing_id = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()

    def windows(self, partition, size):
        for start in range(0, len(self.labels[partition])-size+1, size):
            stop = start+size
            yield Window(self.features[partition][start:stop], self.labels[partition][start:stop],
                         self.raw[partition][start:stop], self.generated[partition][start:stop])


# Historical FPGA asset generation still imports this loader. The simulation uses FlowData.
class StreamingDataset:
    """DRIFTADAPT streaming data with optional per-trace preprocessing segments."""

    def __init__(self, config: dict, base_dir: Path, device: torch.device) -> None:
        self.device = device
        feature_names = [name.strip().lower() for name in config["feature_names"]]
        label_column = config.get("label_column", "label").strip().lower()
        generated_column = config.get("generated_label_column")
        frames: list[pd.DataFrame] = []
        for configured_path in config["files"]:
            path = resolve_path(configured_path, base_dir)
            frame = pd.read_csv(path)
            frame.columns = [str(column).strip().lower() for column in frame.columns]
            frames.append(frame)

        frame = pd.concat(frames, ignore_index=True)
        if "total packet length" in feature_names and "total packet length" not in frame:
            frame["total packet length"] = (
                frame["total length of fwd packets"] + frame["total length of bwd packets"]
            )
        if "number of packets" in feature_names and "number of packets" not in frame:
            frame["number of packets"] = frame["total fwd packets"] + frame["total backward packets"]

        raw = torch.tensor(frame[feature_names].to_numpy(dtype="float32"))
        segments = config.get("preprocessing_segments", [len(current) for current in frames])
        if sum(segments) != len(frame) or any(length <= 0 for length in segments):
            raise ValueError("dataset.preprocessing_segments must be positive and cover every row")
        processed: list[torch.Tensor] = []
        start = 0
        for length in segments:
            features = raw[start:start + length].clone()
            if config.get("standardize", True):
                mean = features.mean(dim=0, keepdim=True)
                std = features.std(dim=0, keepdim=True)
                features = torch.nan_to_num((features - mean) / std)
            if config.get("normalize", False):
                minimum = features.amin(dim=0, keepdim=True)
                span = features.amax(dim=0, keepdim=True) - minimum
                features = torch.nan_to_num((features - minimum) / span)
            processed.append(features)
            start += length

        self.features = torch.cat(processed).to(device)
        self.raw_features = raw.to(device)
        self.labels = torch.tensor(frame[label_column].to_numpy(), dtype=torch.long, device=device)
        self.generated_labels = (
            torch.tensor(frame[generated_column.strip().lower()].to_numpy(), dtype=torch.long, device=device)
            if generated_column else torch.full_like(self.labels, -1)
        )

    def windows(self, window_size: int) -> Iterator[Window]:
        for start in range(0, len(self.labels), window_size):
            end = min(start + window_size, len(self.labels))
            if end - start < window_size:
                break  # artifact only evaluates complete windows
            yield Window(
                self.features[start:end], self.labels[start:end],
                self.raw_features[start:end], self.generated_labels[start:end],
            )


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configured DRIFTADAPT asset does not exist: {path}")
    return path


def benign_classes(config, *, required=False):
    """Explicit anomaly semantics; device identities are not attack labels."""
    classes = config['dataset'].get('benign_classes')
    if classes is None:
        classes = {'unsw-iot': [-1], 'unsw-nb15': [0], 'cic-ids2017': [0]}.get(config['dataset']['name'])
    if classes is None and config['dataset']['num_classes'] == 2:
        classes = [0]
    if classes is None:
        if required:
            raise ValueError('Anomaly methods require dataset.benign_classes for multiclass/device labels; benign identity cannot be inferred')
        return None
    if not classes or any(type(value) is not int or not -1 <= value < config['dataset']['num_classes'] for value in classes):
        raise ValueError('dataset.benign_classes must contain valid class IDs')
    return sorted(set(classes))


class ReleasedFeedback:
    """One explicit label budget and causal release schedule for flow methods.

    JSONL: sample_id (zero-based test row), label (original class ID), verified=true,
    available_window (one-based prediction batch). No labels are read from ground truth.
    Without a file there is no verified feedback. An omitted budget is unlimited;
    teacher requests and released labels both spend the configured max_labels budget.
    """
    def __init__(self, config, sample_count):
        settings = config.get('feedback', {})
        self.budget = settings.get('max_labels')
        if self.budget is not None and (type(self.budget) is not int or self.budget < 0):
            raise ValueError('feedback.max_labels must be a nonnegative integer')
        self.used = self.released = self.cursor = 0
        self.records = []
        size = config['stream']['window_size']
        path = settings.get('file')
        if path:
            seen = set()
            for line in path_for(config, path).read_text().splitlines():
                record = json.loads(line)
                sample = record.get('sample_id')
                available = record.get('available_window')
                label = record.get('label')
                if (type(sample) is not int or not 0 <= sample < sample_count or sample in seen
                    or type(available) is not int or available < sample//size+1
                    or record.get('verified') is not True
                    or type(label) is not int or label not in set(range(config['dataset']['num_classes'])) | set(benign_classes(config) or [])):
                    raise ValueError('Feedback must be unique, verified, in-range, and available only after its flow prediction')
                seen.add(sample)
                self.records.append(record)
            self.records.sort(key=lambda record: record['available_window'])
        self.sample_ids = {record['sample_id'] for record in self.records}

    @property
    def remaining(self):
        return None if self.budget is None else self.budget-self.used

    def release(self, window):
        records = []
        while self.cursor < len(self.records) and self.records[self.cursor]['available_window'] <= window:
            if self.remaining == 0:
                break
            records.append(self.records[self.cursor])
            self.cursor += 1
            self.used += 1
            self.released += 1
        return records
