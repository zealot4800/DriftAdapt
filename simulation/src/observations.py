"""Evaluator-owned paired observations and feature-level synthetic interventions.

These interventions act after packet encoding; they are not physical packet faults.
Give a method only the bound request/advance capabilities, never this service.
"""
from collections import deque
import copy
from numbers import Real
import numpy as np


def _finite(value):
    return isinstance(value, Real) and np.isfinite(value)


class ReferenceChannel:
    def __init__(self, settings=None, seed=0):
        settings = settings or {}
        self.budget = settings.get('budget', 256)
        self.delay = settings.get('delay_windows', 0)
        self.capacity = settings.get('max_pending', 64)
        if any(type(value) is not int or value < minimum for value, minimum in
               ((self.budget, 0), (self.delay, 0), (self.capacity, 1))):
            raise ValueError('Reference budget/delay must be nonnegative integers; max_pending must be positive')
        self.availability = settings.get('availability', 1.)
        self.reference_noise = settings.get('reference_noise', 0.)
        if (not _finite(self.availability) or not 0 <= self.availability <= 1
                or not _finite(self.reference_noise) or self.reference_noise < 0):
            raise ValueError('Reference availability must be in [0,1] and reference_noise nonnegative')
        self.interventions = copy.deepcopy(settings.get('interventions', []))
        if not isinstance(self.interventions, list) or any(not isinstance(item, dict) for item in self.interventions):
            raise ValueError('Reference interventions must be a list of mappings')
        for intervention in self.interventions:
            indices = intervention.get('features', [])
            start, end = intervention.get('start', 0), intervention.get('end')
            if (not isinstance(indices, list) or not indices
                    or any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices)
                    or type(start) is not int or start < 0
                    or end is not None and (type(end) is not int or end < start)):
                raise ValueError('Interventions require unique feature indices and increasing integer start/end')
            if type(intervention.get('ramp', False)) is not bool or intervention.get('ramp') and end is None:
                raise ValueError('Intervention ramp requires a finite end window')
            for key, default in (('scale', 1.), ('shift', 0.), ('noise', 0.), ('step', 0.), ('value', 0.)):
                value = intervention.get(key, default)
                if not _finite(value) or key in ('noise', 'step') and value < 0:
                    raise ValueError(f'Intervention {key} must be finite; noise/step must be nonnegative')
            clip, missing = intervention.get('clip'), intervention.get('missing', 0.)
            if (clip is not None and (not isinstance(clip, (list, tuple)) or len(clip) != 2
                                     or not all(_finite(x) for x in clip) or clip[0] > clip[1])
                    or not _finite(missing) or not 0 <= missing <= 1):
                raise ValueError('Intervention clip must be [lower,upper] and missing must be in [0,1]')
        seeds = np.random.SeedSequence(seed).spawn(3)
        self.rng, self.access_rng, self.reference_rng = (np.random.default_rng(s) for s in seeds)
        self.current, self.sequence, self.requested_current = None, -1, False
        self.pending = deque()
        self.counts = dict(request_attempts=0, requested=0, accepted=0, released=0,
                           denied=0, paired_measurements=0, feature_measurements=0, export_bytes=0)

    def observe(self, identifier, timestamp, reference):
        """Register one current window; return a copy of its operational vector."""
        reference = np.array(reference, dtype=np.float64, copy=True)
        if (type(identifier) is not int or identifier < 0 or not _finite(timestamp)
                or reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all()
                or self.current is not None and identifier <= self.current['id']):
            raise ValueError('Reference observations require increasing integer ids and finite 1D features/time')
        observed = reference.copy()
        for intervention in self.interventions:
            indices = intervention['features']
            if max(indices) >= len(reference):
                raise ValueError('Intervention feature index exceeds the packet feature vector')
            start, end = intervention.get('start', 0), intervention.get('end')
            if identifier < start or end is not None and identifier > end:
                continue
            fraction = (identifier-start)/max(end-start, 1) if intervention.get('ramp') and end > start else 1.
            values = observed[indices].copy()
            values *= 1 + fraction*(intervention.get('scale', 1.)-1)
            values += fraction*intervention.get('shift', 0.)
            values += self.rng.normal(0, fraction*intervention.get('noise', 0.), len(indices))
            step = intervention.get('step', 0.)
            if step:
                values += fraction*(np.round(values/step)*step-values)
            if intervention.get('clip') is not None:
                values += fraction*(np.clip(values, *intervention['clip'])-values)
            missing = self.rng.random(len(indices)) < fraction*intervention.get('missing', 0.)
            values[missing] = intervention.get('value', 0.)
            observed[indices] = values
        with np.errstate(over='ignore', invalid='ignore'):
            observed = observed.astype(np.float32).astype(np.float64)
        if not np.isfinite(observed).all():
            raise ValueError('Observation intervention produced nonfinite features')
        trusted = reference + self.reference_rng.normal(0, self.reference_noise, len(reference))
        if not np.isfinite(trusted).all():
            raise ValueError('Reference stress noise produced nonfinite features')
        self.current = dict(id=identifier, time=float(timestamp), observed=observed.copy(), reference=trusted)
        self.sequence += 1
        self.requested_current = False
        return observed.copy()

    def request(self, identifier, purpose='fit'):
        """Spend one available request token; duplicates/unavailability also cost tokens."""
        self.counts['request_attempts'] += 1
        charged = self.counts['requested'] < self.budget
        self.counts['requested'] += int(charged)
        valid = (self.current is not None and identifier == self.current['id']
                 and purpose in ('fit', 'validation') and not self.requested_current)
        if valid:
            self.requested_current = True
            valid = charged and len(self.pending) < self.capacity and self.access_rng.random() < self.availability
        if not valid:
            self.counts['denied'] += 1
            return False
        record = {key: value.copy() if isinstance(value, np.ndarray) else value
                  for key, value in self.current.items()}
        record['purpose'] = purpose
        self.pending.append((self.sequence+self.delay, record))
        self.counts['accepted'] += 1
        return True

    def advance(self, identifier):
        """Release only eligible queued pairs at the current observation window."""
        if self.current is None or identifier != self.current['id']:
            return []
        released = []
        while self.pending and self.pending[0][0] <= self.sequence:
            _, record = self.pending.popleft()
            released.append(record)
            self.counts['released'] += 1
            self.counts['paired_measurements'] += len(record['reference'])
            self.counts['feature_measurements'] += 2*len(record['reference'])
            self.counts['export_bytes'] += record['reference'].nbytes + record['observed'].nbytes
        return released

    def stats(self):
        return {**self.counts, 'pending': len(self.pending), 'remaining': self.budget-self.counts['requested']}
