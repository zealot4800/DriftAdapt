"""Reference-relative, selective LUT repair; the boot MLP and threshold stay frozen."""
from collections import deque
import math
import numpy as np
import torch
from torch.nn import functional as F
from ..metrics import calibrate_threshold
from ..models import MLP, fit_model


class DriftAdapt:
    unit = 'detection_window'
    defaults = dict(fit_pairs=32, min_support=4, max_entries=32, max_proposals=128,
                    validation_blocks=48, block_windows=4, validation_timeout=128,
                    epsilon=.1, confidence=.05, margin_weight=.01,
                    entry_penalty=.0001, magnitude_penalty=.00001,
                    max_fpr=.15, distillation_weight=.1, policy='recoverability')

    def __init__(self, config):
        self.settings = {key: config.get('driftadapt', {}).get(key, value) for key, value in self.defaults.items()}
        for key, default in self.defaults.items():
            value = self.settings[key]
            if key == 'policy':
                if value not in ('uniform', 'decision', 'recoverability'):
                    raise ValueError('DriftAdapt policy must be uniform, decision or recoverability')
            elif (not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                  or type(default) is int and type(value) is not int):
                raise ValueError(f'driftadapt.{key} must be positive and finite')
        if any(self.settings[key] >= 1 for key in ('epsilon', 'confidence', 'max_fpr')):
            raise ValueError('DriftAdapt epsilon, confidence and max_fpr must be below one')
        self.training = dict(config['training'])
        self.device = torch.device(config['runtime']['device'])
        self.rng = np.random.default_rng(config['runtime']['seed'])
        self.fit_pairs = deque(maxlen=self.settings['fit_pairs'])
        self.version = self.attempts = self.rejected = self.alarms = self.recoveries = 0
        self.candidate = None
        self.status, self.alarm = 'uncertain', False
        self.diagnostics = {}
        self.contract = {}

    def _indices(self, values):
        return np.floor(np.clip((np.asarray(values)-self.minimum)/self.span, 0, 1)*64).clip(0, 63).astype(np.int64)

    def _encoded(self, values, table, indices=None):
        if indices is None:
            indices = self._indices(values)
        return torch.from_numpy(table[np.arange(self.features), indices]).to(self.device)

    @torch.inference_mode()
    def _scores(self, values, table, indices=None):
        logits = self.model(self._encoded(values, table, indices))
        return (logits[:, 1]-logits[:, 0]).cpu().numpy().astype(np.float64)

    def train(self, data):
        source, offline = (values.cpu().numpy().astype(np.float64) for values in data.features[:2])
        labels, offline_labels = (values.cpu().numpy() for values in data.labels[:2])
        if any(set(np.unique(y)) != {0, 1} for y in (labels, offline_labels)):
            raise ValueError('DriftAdapt requires both classes in source and offline packet windows')
        self.features = source.shape[1]
        self.minimum, self.span = source.min(axis=0), np.ptp(source, axis=0)
        self.span = np.where(self.span > 0, self.span, 1.)
        bins = self._indices(source)
        counts = np.array([[np.bincount(bins[labels == label, f], minlength=64)
                            for f in range(self.features)] for label in (0, 1)])
        probabilities = (counts+1)/(counts.sum(axis=2, keepdims=True)+64)
        self.boot = np.log(probabilities[1]/probabilities[0]).astype(np.float32)
        self.boot.setflags(write=False)
        self.table = self.boot.copy()
        self.model = MLP(self.features).to(self.device)
        encoded = self._encoded(source, self.boot)
        targets = torch.stack((torch.as_tensor(labels, device=self.device), encoded.sum(1)), dim=1)
        def loss(logits, target):
            return (F.cross_entropy(logits, target[:, 0].long())+self.settings['distillation_weight']*
                    F.smooth_l1_loss(logits[:, 1]-logits[:, 0], target[:, 1]))
        fit_model(self.model, encoded, targets, self.training, loss=loss)
        self.model.requires_grad_(False)
        scores = self._scores(offline, self.boot)
        self.threshold = calibrate_threshold(scores, offline_labels, self.settings['max_fpr'])
        self.frozen = tuple(p.detach().clone() for p in self.model.parameters())
        self.boot_threshold = self.threshold
        self.margin_scale = max(float(np.std(scores)), 1e-6)
        self.support = np.zeros_like(self.boot, dtype=np.int64)

    def bind_reference(self, request, receive, telemetry):
        """Only capabilities are retained; no simulator, fault settings or target dataset."""
        self.request, self.receive, self.telemetry = request, receive, telemetry

    def infer(self, observation):
        values = observation['features'].detach().cpu().numpy()
        return (self._scores(values, self.table) >= self.threshold).astype(np.int64)

    def _localize(self, observed, reference):
        z, x = self._indices(observed), self._indices(reference)
        decisions = self._scores(reference, self.boot) >= self.threshold
        _, inverse = np.unique(z, axis=0, return_inverse=True)
        collisions = np.array([np.bincount(decisions[inverse == group].astype(int), minlength=2)
                               for group in range(inverse.max()+1)])
        self.diagnostics = {'empirical_collision_error': float(collisions.min(axis=1).sum()/len(z)),
                            'ambiguous_regions': int((collisions.min(axis=1) > 0).sum()),
                            'singleton_regions': int((collisions.sum(axis=1) == 1).sum()),
                            'features': {}}
        encoded = self._encoded(reference, self.boot).requires_grad_(True)
        logits = self.model(encoded)
        sensitivity = torch.autograd.grad((logits[:, 1]-logits[:, 0]).sum(), encoded)[0].abs().mean(0).cpu().numpy()
        proposals = []
        affected = (z != x).any(axis=0)
        for feature in np.flatnonzero(affected | np.any(self.table != self.boot, axis=1)):
            counts = np.zeros((64, 64), dtype=np.int64)
            np.add.at(counts, (z[:, feature], x[:, feature]), 1)
            occupied = counts.sum(axis=1) > 0
            multiplicity = (counts > 0).sum(axis=1)
            delta = observed[:, feature]-reference[:, feature]
            centered = reference[:, feature]-reference[:, feature].mean()
            scale = float(centered @ (observed[:, feature]-observed[:, feature].mean())/(centered @ centered)) if centered.any() else None
            residual = delta if scale is None else observed[:, feature]-scale*reference[:, feature]
            if affected[feature]:
                self.diagnostics['features'][int(feature)] = {
                    'displacement': float(np.median(delta)), 'scale_estimate': scale,
                    'noise_mad': float(np.median(np.abs(residual-np.median(residual)))),
                    'decision_sensitivity': float(sensitivity[feature]),
                    'support': int((z[:, feature] != x[:, feature]).sum()),
                    'reference_bins': int((counts.sum(axis=0) > 0).sum()),
                    'mapping_evidence': 'empirical_only',
                    'many_to_one_bins': int((multiplicity > 1).sum()),
                    'constant_observation': bool(np.ptp(observed[:, feature]) == 0)}
            for value in np.flatnonzero(occupied):
                row = counts[value]
                if row.sum() < self.settings['min_support']:
                    continue
                if row[value] == row.sum():
                    target = float(self.boot[feature, value])  # Supported restoration, including rollback.
                elif row[value] or multiplicity[value] != 1:
                    continue  # Preserve identity/ambiguous regions; more evidence cannot be invented.
                else:
                    target = float(self.boot[feature, row.argmax()])
                if target != self.table[feature, value]:
                    proposals.append((int(feature), int(value), target))
        return proposals

    def _loss(self, scores, reference_scores, table):
        disagreement = np.mean((scores >= self.threshold) != (reference_scores >= self.threshold))
        margin = np.mean(np.minimum(np.abs(scores-reference_scores)/self.margin_scale, 1.))
        changed = table != self.boot
        return (disagreement+self.settings['margin_weight']*margin+
                self.settings['entry_penalty']*changed.sum()+
                self.settings['magnitude_penalty']*np.abs(table-self.boot).sum())

    def _propose(self):
        observed = np.stack([p['observed'] for p in self.fit_pairs])
        reference = np.stack([p['reference'] for p in self.fit_pairs])
        reference_scores = self._scores(reference, self.boot)
        indices = self._indices(observed)
        current_scores = self._scores(observed, self.table, indices)
        before = float(np.mean((current_scores >= self.threshold) != (reference_scores >= self.threshold)))
        proposals = self._localize(observed, reference)
        relationship_changed = bool(self.diagnostics['features'])
        self.alarms += int(relationship_changed and not self.alarm)
        self.recoveries += int(not relationship_changed and self.alarm)
        self.alarm = relationship_changed
        if before == 0:
            self.status = 'observation_change' if relationship_changed else 'healthy'
            return None
        if self.diagnostics['empirical_collision_error'] > self.settings['epsilon']:
            self.status = 'information_loss_evidence'
            return None
        candidate = self.table.copy()
        initial_loss = baseline = self._loss(current_scores, reference_scores, candidate)
        # Reuse bins and table storage; keep the same candidate order and NN batch shape.
        ranked = []
        for feature, value, target in proposals:
            previous = candidate[feature, value]
            candidate[feature, value] = target
            if self._verify(candidate):
                loss = self._loss(self._scores(observed, candidate, indices), reference_scores, candidate)
                ranked.append((loss, feature, value, target))
            candidate[feature, value] = previous
        remaining = sorted(ranked)[:self.settings['max_proposals']]
        for _ in range(self.settings['max_entries']):
            best, selected = baseline, None
            for _, feature, value, target in remaining:
                previous = candidate[feature, value]
                candidate[feature, value] = target
                if self._verify(candidate):
                    loss = self._loss(self._scores(observed, candidate, indices), reference_scores, candidate)
                    if loss < best-1e-12:
                        best, selected = loss, (feature, value, target)
                candidate[feature, value] = previous
            if selected is None:
                break
            feature, value, target = selected
            candidate[feature, value], baseline = target, best
            remaining = [p for p in remaining if p[1:3] != (feature, value)]
        if np.array_equal(candidate, self.table) or not self._verify(candidate):
            self.status = 'insufficient_repair_support'
            return None
        if baseline >= initial_loss:
            self.status = 'no_supported_improvement'
            return None
        return candidate

    def _verify(self, table):
        return (table.shape == self.boot.shape and table.dtype == self.boot.dtype and np.isfinite(table).all()
                and np.count_nonzero(table != self.boot) <= self.settings['max_entries']
                and np.count_nonzero(table != self.table) <= self.settings['max_entries']
                and self.threshold == self.boot_threshold
                and all(torch.equal(p, frozen) for p, frozen in zip(self.model.parameters(), self.frozen)))

    def _validate(self):
        size = self.settings['block_windows']
        records = self.validation
        blocks = [[records.get(i) for i in range(start, start+size)]
                  for start in range(self.validation_start, self.validation_end+1, size)]
        blocks = [block for block in blocks if all(p is not None for p in block)]
        failures, before, after, count = 0, 0, 0, 0
        for block in blocks:
            observed = np.stack([p['observed'] for p in block])
            reference = np.stack([p['reference'] for p in block])
            target = self._scores(reference, self.boot) >= self.threshold
            old = self._scores(observed, self.table) >= self.threshold
            new = self._scores(observed, self.candidate) >= self.threshold
            failures += int(np.any(new != target))
            before += int(np.count_nonzero(old != target))
            after += int(np.count_nonzero(new != target))
            count += len(block)
        alpha = self.settings['confidence']/(self.attempts*(self.attempts+1))
        # Exponential-supermartingale bound: dependence inside/across blocks is allowed.
        rate = math.log(100.)
        upper = min(1., (rate*failures+math.log(1/alpha))/(len(blocks)*.99)) if blocks else 1.
        self.contract = {'disagreement_upper': upper, 'validation_blocks': len(blocks),
                         'before': before/count if count else None, 'after': after/count if count else None,
                         'alpha': alpha, 'window_range': [self.validation_start, self.validation_end],
                         'scope': 'validation_cohort_conditional_block_risk'}
        if (len(blocks) == self.validation_blocks and upper <= self.settings['epsilon']
                and after < before and self._verify(self.candidate)):
            previous = self.table
            self.table, self.version, self.status = self.candidate, self.version+1, 'repaired'
            delta = self.table-self.boot
            self.contract.update(changed_entries=int(np.count_nonzero(delta)),
                                 changed_tables=int(np.count_nonzero(np.any(delta != 0, axis=1))),
                                 modified_words=int(np.count_nonzero(self.table != previous)),
                                 maximum_change=float(np.abs(delta).max()), total_change=float(np.abs(delta).sum()))
        else:
            self.rejected += 1
            self.status = ('validation_rejected' if len(blocks) == self.validation_blocks
                           else 'insufficient_validation')
        self.candidate = None
        self.fit_pairs.clear()

    def adapt(self, observation, prediction):
        identifier = observation['id']
        def collect():
            for pair in self.receive(identifier):
                if pair['purpose'] == 'fit' and self.candidate is None:
                    self.fit_pairs.append(pair)
                    indices = self._indices(pair['observed'])
                    self.support[np.arange(self.features), indices] += 1
                elif (pair['purpose'] == 'validation' and self.candidate is not None
                      and self.validation_start <= pair['id'] <= self.validation_end):
                    self.validation[pair['id']] = pair
        collect()
        if self.candidate is not None:
            if identifier <= self.validation_end:
                self.request(identifier, 'validation')
            collect()
            if identifier >= self.validation_end and (not self.telemetry()['pending'] or
                    identifier >= self.validation_end+self.settings['validation_timeout']):
                self._validate()
            return
        values = observation['features'].detach().cpu().numpy()
        margin = abs(float(self._scores(values, self.table)[0]-self.threshold))/self.margin_scale
        indices = self._indices(values)[0]
        policy = self.settings['policy']
        chance = 1. if policy == 'uniform' else .1+.9/(1+margin)
        if policy == 'recoverability':
            chance *= float(np.mean(1/np.sqrt(1+self.support[np.arange(self.features), indices])))
        if self.rng.random() < chance:
            self.request(identifier, 'fit')
        collect()
        if len(self.fit_pairs) == self.settings['fit_pairs']:
            self.candidate = self._propose()
            self.fit_pairs.clear()
            if self.candidate is not None:
                self.attempts += 1
                self.validation_start = identifier+1
                alpha = self.settings['confidence']/(self.attempts*(self.attempts+1))
                self.validation_blocks = max(self.settings['validation_blocks'],
                    math.ceil(math.log(1/alpha)/(.99*self.settings['epsilon'])))
                self.validation_end = identifier+self.validation_blocks*self.settings['block_windows']
                self.validation, self.status = {}, 'validating'
        if not self.telemetry()['remaining'] and not self.telemetry()['pending'] and self.candidate is None:
            self.status = 'budget_exhausted'

    def state(self):
        telemetry = self.telemetry() if hasattr(self, 'telemetry') else {}
        return {'adaptation_count': self.version, 'labeling_cost_samples': 0, 'repair_status': self.status,
                'repair_attempts': self.attempts, 'rejected_candidates': self.rejected,
                'observation_alarms': self.alarms, 'observation_recoveries': self.recoveries,
                'reference': telemetry, 'contract': self.contract, 'recoverability': self.diagnostics,
                'validation_pairs': len(self.validation) if self.candidate is not None else 0}
