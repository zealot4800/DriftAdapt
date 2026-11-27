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


def detection_metrics(expected, predicted):
    expected, predicted = np.asarray(expected), np.asarray(predicted)
    if not len(expected):
        return None
    result = {'accuracy':float(np.mean(expected == predicted)),
              'macro_f1':macro_f1(torch.as_tensor(expected), torch.as_tensor(predicted)),
              'labeled_predictions':len(expected)}
    benign = 0
    tp = int(((expected == 1) & (predicted == 1)).sum())
    fp = int(((expected == benign) & (predicted == 1)).sum())
    fn = int(((expected == 1) & (predicted == benign)).sum())
    tn = int(((expected == benign) & (predicted == benign)).sum())
    result.update(tp=tp, fp=fp, fn=fn, tn=tn, benign_samples=tn+fp, attack_samples=tp+fn,
                  both_classes_present=bool(tn+fp and tp+fn),
                  anomaly_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                  false_positive_rate=fp/(fp+tn) if fp+tn else None,
                  recall=tp/(tp+fn) if tp+fn else None)
    return result


def calibrate_threshold(scores, labels, max_fpr):
    """Best offline macro-F1 threshold satisfying the benign FPR constraint."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Threshold calibration requires finite scores and both offline classes')
    order = np.argsort(scores, kind='stable')
    sorted_scores, sorted_labels = scores[order], labels[order]
    thresholds, first = np.unique(sorted_scores, return_index=True)
    benign = (sorted_labels == 0).astype(np.int64)
    attacks = 1-benign
    fp = benign.sum()-np.concatenate(([0], np.cumsum(benign)))[first]
    tp = attacks.sum()-np.concatenate(([0], np.cumsum(attacks)))[first]
    fn, tn = attacks.sum()-tp, benign.sum()-fp
    quality = .5*(2*tp/np.maximum(2*tp+fp+fn, 1)+2*tn/np.maximum(2*tn+fp+fn, 1))
    thresholds = np.append(thresholds, np.nextafter(thresholds[-1], np.inf))
    quality = np.append(quality, benign.sum()/(2*benign.sum()+attacks.sum()))
    valid = np.append(fp/benign.sum() <= max_fpr, True)
    return float(thresholds[np.argmax(np.where(valid, quality, -np.inf))])
