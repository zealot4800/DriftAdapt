"""DriftAdapt: distilled encoded MLP with histogram monitoring and verified feedback."""
import copy
from collections import deque

import numpy as np
import torch
from torch import nn

from ..config import path_for
from ..data import benign_classes, frame_fingerprint
from ..metrics import macro_f1
from ..models import retrain_model


def _fake_quantize(value, scale, minimum, maximum):
    quantized = (value / scale).round().clamp(minimum, maximum) * scale
    return value + (quantized - value).detach()


class EvidenceDistillationLoss(nn.Module):
    def __init__(self, benign_cost, strength):
        super().__init__()
        self.register_buffer('class_weights', torch.tensor([benign_cost, 1.0]))
        self.strength = strength

    def forward(self, logits, target):
        known = target[:, 0] >= 0
        supervised = (nn.functional.cross_entropy(logits[known], target[known, 0].long(),
                      weight=self.class_weights) if known.any() else logits.sum() * 0)
        teacher = nn.functional.smooth_l1_loss((logits[:, 1] - logits[:, 0]) / 4, target[:, 1] / 4)
        return supervised + self.strength * teacher


class CompactMLP(nn.Module):
    def __init__(self, input_bound):
        super().__init__()
        self.register_buffer('input_bound', torch.as_tensor(input_bound, dtype=torch.float32))
        self.register_buffer('hidden_scales', torch.ones(2))
        sizes = (len(input_bound), 16, 8, 2)
        self.layers = nn.ModuleList(nn.Linear(left, right) for left, right in zip(sizes, sizes[1:]))
        self.qat = False

    def forward(self, features):
        x = (features / self.input_bound).clamp(-1, 1)
        scale = 1 / 127
        if self.qat:
            x = _fake_quantize(x, scale, -127, 127)
        for index, layer in enumerate(self.layers):
            weight, bias = layer.weight, layer.bias
            if self.qat:
                weight_scale = weight.detach().abs().max().clamp_min(1e-8) / 127
                weight = _fake_quantize(weight, weight_scale, -127, 127)
                bias = _fake_quantize(bias, scale * weight_scale, -(2**31), 2**31 - 1)
            x = nn.functional.linear(x, weight, bias)
            if index < 2:
                x = x.relu()
                scale = self.hidden_scales[index]
                if self.qat:
                    x = _fake_quantize(x, scale, 0, 255)
        return x

    @torch.no_grad()
    def calibrate_activations(self, source):
        x = (source / self.input_bound).clamp(-1, 1)
        for index, layer in enumerate(self.layers[:2]):
            x = layer(x).relu()
            self.hidden_scales[index] = x.max().clamp_min(1e-6) / 255


class DriftAdapt:
    unit = 'flow'
    evaluation_target = 'anomaly'
    architecture = 'distilled_evidence_quantized_mlp_v4'
    encoder_bins = 64
    requant_shift = 24

    def __init__(self, config, output):
        self.config = config
        self.benign = benign_classes(config, required=True)
        self.max_fpr = float(config.get('driftadapt', {}).get('max_false_positive_rate', 0.15))
        if not 0 <= self.max_fpr <= 1:
            raise ValueError('driftadapt.max_false_positive_rate must be between zero and one')
        self.benign_cost = 1.0
        self.distillation_weight = 0.0
        self.checkpoint = (path_for(config, config['output']['directory']) / 'artifacts'
                           / f"{config['dataset']['name']}-driftadapt-mlp-v4.pt")
        self.pending = {}
        self.feedback = deque(maxlen=2048)
        self.version = self.update_bytes = self.feedback_count = self.new_feedback = self.streak = 0

    def _schema(self, data):
        return {'architecture': self.architecture, 'preprocessing_id': data.preprocessing_id,
                'offline_sha256': frame_fingerprint(data.frames[1]), 'benign_classes': self.benign,
                'seed': self.config['runtime']['seed'], 'training': dict(self.config['training']),
                'max_false_positive_rate': self.max_fpr, 'encoder_bins': self.encoder_bins}

    def _labels(self, labels):
        return (~np.isin(labels, self.benign)).astype(np.int64)

    def _indices(self, features):
        return np.clip(np.floor((features - self.minimum) * self.inverse_width),
                       0, self.bins - 1).astype(np.int64)

    @staticmethod
    def _counts(indices, bins):
        offsets = np.arange(indices.shape[1]) * bins
        return np.bincount((indices + offsets).ravel(),
                           minlength=indices.shape[1] * bins).reshape(-1, bins)

    def _encoder_indices(self, features):
        return np.clip(np.floor((features - self.encoder_minimum) * self.encoder_inverse_width),
                       0, self.encoder_bins - 1).astype(np.int64)

    def _encode(self, features):
        return self.encoder_table[np.arange(features.shape[1]), self._encoder_indices(features)]

    def _fit_encoder(self, features, labels):
        self.encoder_minimum = features.min(axis=0)
        span = np.ptp(features, axis=0)
        self.encoder_inverse_width = self.encoder_bins / np.where(span > 0, span, 1)
        indices = self._encoder_indices(features)
        counts = np.stack([self._counts(indices[labels == label], self.encoder_bins) for label in (0, 1)])
        probability = (counts + 1) / (counts.sum(axis=2, keepdims=True) + self.encoder_bins)
        self.encoder_table = np.log(probability[1] / probability[0]).astype(np.float32)

    @torch.no_grad()
    def _export(self):
        self.packed = []
        scale = 1 / 127
        for index, layer in enumerate(self.model.layers):
            weight = layer.weight.detach().cpu().numpy()
            weight_scale = max(float(np.abs(weight).max()), 1e-8) / 127
            quantized = np.clip(np.rint(weight / weight_scale), -127, 127).astype(np.int8)
            bias = np.clip(np.rint(layer.bias.detach().cpu().numpy().astype(np.float64) / (scale * weight_scale)),
                           -(2**31), 2**31 - 1).astype(np.int32)
            multiplier = None
            if index < 2:
                output_scale = float(self.model.hidden_scales[index])
                multiplier = round(scale * weight_scale / output_scale * (1 << self.requant_shift))
                if not 0 <= multiplier < 2**31:
                    raise ValueError('MLP requantization multiplier exceeds the signed 32-bit export range')
                scale = output_scale
            self.packed.append((quantized, bias, multiplier))
        input_factor = 127 / self.model.input_bound.detach().cpu().numpy()
        self.input_table = np.clip(np.rint(self.encoder_table * input_factor[:, None]),
                                   -127, 127).astype(np.int8)
        self.parameter_bytes = (sum(w.nbytes + b.nbytes + (4 if m is not None else 0)
                                   for w, b, m in self.packed) + self.input_table.nbytes
                                + self.encoder_minimum.nbytes + self.encoder_inverse_width.nbytes + 4)
        self.macs = sum(w.size for w, _, _ in self.packed)

    def _scores(self, features):
        x = self.input_table[np.arange(features.shape[1]), self._encoder_indices(features)].astype(np.int32)
        for index, (weight, bias, multiplier) in enumerate(self.packed):
            x = x @ weight.astype(np.int64).T + bias
            if np.any((x < -(2**31)) | (x >= 2**31)):
                raise ValueError('MLP accumulation exceeds the signed 32-bit export range')
            if index < 2:
                x = np.clip((x.clip(0) * multiplier + (1 << (self.requant_shift - 1)))
                            >> self.requant_shift, 0, 255).astype(np.int32)
        margin = x[:, 1] - x[:, 0]
        if np.any((margin < -(2**31)) | (margin >= 2**31 - 1)):
            raise ValueError('MLP decision margin exceeds the signed 32-bit export range')
        return margin.astype(np.int32)

    def _threshold(self, scores, labels):
        order = np.argsort(scores, kind='stable')
        values = np.unique(scores)
        thresholds = np.r_[values, values[-1] + 1]
        below = np.searchsorted(scores[order], thresholds)
        positives = np.r_[0, np.cumsum(labels[order])]
        tp, tn = labels.sum() - positives[below], below - positives[below]
        errors = len(labels) - tp - tn
        macro = (2 * tp / np.maximum(2 * tp + errors, 1)
                 + 2 * tn / np.maximum(2 * tn + errors, 1)) / 2
        negatives = int((labels == 0).sum())
        macro[negatives - tn > self.max_fpr * negatives] = -np.inf
        return int(thresholds[macro.argmax()])

    @staticmethod
    def _rates(scores, labels, threshold):
        prediction = scores >= threshold
        return {'macro_f1': macro_f1(torch.as_tensor(labels), torch.as_tensor(prediction)),
                'false_positive_rate': float(prediction[labels == 0].mean()),
                'recall': float(prediction[labels == 1].mean())}

    def _fit(self, encoded, labels, epochs, optimizer=None, distill=True):
        settings = self.config['training']
        criterion = None
        if distill and self.distillation_weight > 0:
            extra = self.distillation_features
            encoded = torch.cat((encoded, extra))
            targets = torch.cat((labels, labels.new_full((len(extra),), -1))).float()
            labels = torch.stack((targets, encoded.sum(dim=1)), dim=1)
            criterion = EvidenceDistillationLoss(self.benign_cost, self.distillation_weight).to(encoded.device)
        return retrain_model(self.model, encoded, labels, epochs=epochs,
            batch_size=settings['batch_size'], learning_rate=settings['learning_rate'],
            optimizer_name=settings['optimizer'], optimizer=optimizer,
            class_weights=torch.tensor([self.benign_cost, 1.0], device=encoded.device),
            criterion=criterion)[1]

    def _load(self, data):
        saved = torch.load(self.checkpoint, map_location=data.device, weights_only=True)
        if saved['schema'] != self._schema(data):
            raise ValueError('MLP checkpoint source, offline partition, or training settings differ from runtime')
        self.model = CompactMLP(torch.ones(len(data.feature_names))).to(data.device)
        self.model.load_state_dict(saved['state_dict'])
        self.encoder_minimum = saved['encoder_minimum'].cpu().numpy()
        self.encoder_inverse_width = saved['encoder_inverse_width'].cpu().numpy()
        self.encoder_table = saved['encoder_table'].cpu().numpy()
        self.model.qat = True
        self.model.eval()
        self.threshold = saved['threshold']
        self.benign_cost = saved['training']['benign_loss_weight']
        self.distillation_weight = saved['training']['distillation_weight']
        self._export()
        return saved['training']

    def train(self, data):
        if self.checkpoint.exists():
            return dict(self._load(data), reused_checkpoint=str(self.checkpoint))
        source = data.features[0]
        labels = self._labels(data.labels[0].cpu().numpy())
        source_labels = torch.as_tensor(labels, device=data.device)
        offline = data.features[1].cpu().numpy()
        target = self._labels(data.labels[1].cpu().numpy())
        cut = len(target) // 2
        if any(len(np.unique(part)) != 2 for part in (labels, target[:cut], target[cut:])):
            raise ValueError('DriftAdapt needs both classes in source and each chronological offline half')
        self._fit_encoder(source.cpu().numpy(), labels)
        bound = np.maximum(np.abs(self.encoder_table).max(axis=1), 1e-6)
        encoded_source = torch.as_tensor(self._encode(source.cpu().numpy()), device=data.device)
        epochs = int(self.config['training']['epochs'])
        if epochs < 1:
            raise ValueError('MLP training epochs must be positive')
        best, candidates = None, []
        self.distillation_features = torch.as_tensor(self._encode(offline[:cut]), device=data.device)

        def consider(trained, baseline=None):
            nonlocal best
            self._export()
            scores = self._scores(offline)
            threshold = self._threshold(scores[:cut], target[:cut])
            calibration = self._rates(scores[:cut], target[:cut], threshold)
            validation = self._rates(scores[cut:], target[cut:], threshold)
            eligible = (baseline is None or
                        (validation['false_positive_rate'] <= baseline['false_positive_rate']
                         and validation['recall'] >= baseline['recall']))
            rank = (validation['macro_f1'], -validation['false_positive_rate'], validation['recall'])
            candidate = {'epochs': trained, 'qat_epochs': trained - epochs,
                'benign_loss_weight': self.benign_cost, 'distillation_weight': self.distillation_weight,
                'calibration': calibration, 'validation': validation,
                'validation_fpr_target_met': validation['false_positive_rate'] <= self.max_fpr,
                'eligible': eligible, 'threshold': threshold}
            candidates.append(candidate)
            if eligible and (best is None or rank > best[0]):
                best = rank, copy.deepcopy(self.model.state_dict()), candidate

        baseline = None
        for distilling in (False, True):
            plans = ([(best[2]['benign_loss_weight'], strength) for strength in (0.25, 1.0, 4.0)]
                     if distilling else [(cost, 0.0) for cost in (1.0, 2.0, 4.0)])
            for cost, strength in plans:
                torch.manual_seed(self.config['runtime']['seed'])
                self.model = CompactMLP(bound).to(data.device)
                self.benign_cost, self.distillation_weight = cost, 0.0
                optimizer = self._fit(encoded_source, source_labels, epochs)
                self.model.calibrate_activations(encoded_source)
                self.model.qat = True
                if not distilling:
                    consider(epochs)
                self.distillation_weight = strength
                trained = epochs
                for total_epochs in (2 * epochs, 4 * epochs):
                    optimizer = self._fit(encoded_source, source_labels, total_epochs - trained, optimizer)
                    trained = total_epochs
                    consider(trained, baseline)
            if not distilling:
                baseline = dict(best[2]['validation'])
        _, state, selected = best
        self.model.load_state_dict(state)
        self.model.eval()
        self.threshold, self.benign_cost = selected['threshold'], selected['benign_loss_weight']
        self.distillation_weight = selected['distillation_weight']
        self._export()
        report = {'architecture': self.architecture, 'layers': [len(bound), 16, 8, 2],
                  'source_training_samples': len(labels), 'offline_calibration_samples': cut,
                  'offline_selection_samples': len(target) - cut, 'candidates': candidates,
                  'selected_epochs': selected['epochs'], 'benign_loss_weight': self.benign_cost,
                  'distillation_weight': self.distillation_weight, 'baseline_validation': baseline,
                  'unlabeled_distillation_samples': cut if self.distillation_weight > 0 else 0,
                  'max_false_positive_rate': self.max_fpr, 'calibration': selected['calibration'],
                  'validation': selected['validation'],
                  'validation_fpr_target_met': selected['validation_fpr_target_met'],
                  'threshold': self.threshold, 'weight_bits': 8, 'bias_bits': 32,
                  'encoder': 'source_class_histogram_log_ratio', 'encoder_bins': self.encoder_bins,
                  'encoder_table_bits': 8, 'selection_objective': 'validation_macro_f1'}
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'schema': self._schema(data), 'state_dict': self.model.state_dict(),
                    'threshold': self.threshold, 'training': report,
                    'encoder_minimum': torch.from_numpy(self.encoder_minimum),
                    'encoder_inverse_width': torch.from_numpy(self.encoder_inverse_width),
                    'encoder_table': torch.from_numpy(self.encoder_table)}, self.checkpoint)
        return report

    def prepare(self, data):
        self._load(data)
        source = data.features[0].cpu().numpy()
        self.source = torch.as_tensor(self._encode(source), device=data.device)
        self.source_labels = torch.as_tensor(self._labels(data.labels[0].cpu().numpy()), device=data.device)
        self.bins = int(self.config['drift']['bins'])
        self.minimum = source.min(axis=0)
        span = np.ptp(source, axis=0)
        self.inverse_width = self.bins / np.where(span > 0, span, 1)
        self.reference = self._counts(self._indices(source), self.bins)
        self.reference_samples = len(source)
        cut = len(data.labels[1]) // 2
        self.calibration = data.features[1][:cut].cpu().numpy()
        self.distillation_features = torch.as_tensor(self._encode(self.calibration), device=data.device)
        self.calibration_labels = self._labels(data.labels[1][:cut].cpu().numpy())

    def infer(self, observation):
        self.features = observation['features'].cpu().numpy()
        return (self._scores(self.features) >= self.threshold).astype(np.int64)

    def adapt(self, observation, prediction):
        current = self._counts(self._indices(self.features), self.bins)
        samples = len(self.features)
        difference = np.abs(current * self.reference_samples - self.reference * samples).sum(axis=1)
        threshold = self.config['drift']['threshold']
        shifted = bool(np.any(difference > 2 * threshold * samples * self.reference_samples))
        self.streak = self.streak + 1 if shifted else 0
        active = self.streak >= self.config['drift']['consecutive_windows']
        for offset, sample_id in enumerate(observation['sample_ids']):
            if sample_id in observation['feedback_sample_ids']:
                self.pending[sample_id] = self.features[offset].copy()
        for record in observation['feedback']:
            self.feedback.append((self.pending.pop(record['sample_id']), int(record['label'] not in self.benign)))
        released = len(observation['feedback'])
        self.feedback_count += released
        self.new_feedback += released
        adapted = self.new_feedback >= (8 if active else 32)
        size = 0
        if adapted:
            features, labels = zip(*self.feedback)
            device = self.source.device
            x = torch.cat((self.source, torch.as_tensor(self._encode(np.stack(features)), device=device)))
            y = torch.cat((self.source_labels, torch.as_tensor(labels, device=device)))
            torch.manual_seed(self.config['runtime']['seed'] + self.version + 1)
            self._fit(x, y, max(1, self.config['training']['epochs'] // 3), distill=False)
            self._export()
            self.threshold = self._threshold(self._scores(self.calibration), self.calibration_labels)
            size = self.parameter_bytes
            self.version += 1
            self.update_bytes += size
            self.new_feedback = 0
        return {'adapted': adapted, 'model_update_bytes': size, 'verified_feedback_applied': released,
                'feature_drift_detected': active, 'drift_metric': 'total_variation', 'drift_threshold': threshold}

    def state(self):
        return {'architecture': self.architecture, 'adaptation_count': self.version,
                'labeling_cost_samples': self.feedback_count, 'labeling_seconds': None,
                'model_update_bytes': self.update_bytes, 'layers': [len(self.encoder_table), 16, 8, 2],
                'weight_bits': 8, 'bias_bits': 32, 'activation_bits': 8,
                'score_accumulator_bits': 32, 'requantization_intermediate_bits': 64,
                'inference_parameter_bytes': self.parameter_bytes,
                'multiply_accumulates_per_flow': self.macs,
                'input_encoding': 'source_class_histogram_log_ratio',
                'encoder_bins': self.encoder_bins, 'encoder_table_bytes': self.input_table.nbytes,
                'distillation_weight': self.distillation_weight,
                'decision_threshold': self.threshold, 'histogram_bins': self.bins,
                'calibration_fpr_target': self.max_fpr, 'benign_loss_weight': self.benign_cost}
