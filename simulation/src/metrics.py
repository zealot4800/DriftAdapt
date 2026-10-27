from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score


def macro_f1(expected: torch.Tensor, predicted: torch.Tensor) -> float:
    if expected.numel() == 0:
        return float("nan")
    return float(f1_score(
        expected.detach().cpu().numpy(), predicted.detach().cpu().numpy(),
        average="macro", zero_division=0,
    ))


def write_json(path, value):
    import json
    import math
    from pathlib import Path
    def clean(item):
        if isinstance(item, dict):
            return {key:clean(value) for key,value in item.items()}
        if isinstance(item, (tuple,list)):
            return [clean(value) for value in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')


def detection_metrics(expected, predicted, unit):
    expected, predicted = np.asarray(expected), np.asarray(predicted)
    if not len(expected):
        return None
    result = {'accuracy':float(np.mean(expected == predicted)),
              'macro_f1':macro_f1(torch.as_tensor(expected), torch.as_tensor(predicted)),
              'labeled_predictions':len(expected)}
    if unit in ('detection_window', 'binary_anomaly'):
        benign = -1 if unit == 'detection_window' else 0
        tp = int(((expected == 1) & (predicted == 1)).sum())
        fp = int(((expected == benign) & (predicted == 1)).sum())
        fn = int(((expected == 1) & (predicted == benign)).sum())
        tn = int(((expected == benign) & (predicted == benign)).sum())
        result.update(tp=tp, fp=fp, fn=fn, tn=tn, benign_samples=tn+fp, attack_samples=tp+fn,
                      both_classes_present=bool(tn+fp and tp+fn),
                      anomaly_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                      false_positive_rate=fp/(fp+tn) if fp+tn else None)
    return result
