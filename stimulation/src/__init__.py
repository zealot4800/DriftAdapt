"""Shared architecture selection, with model dependencies loaded on demand."""
from copy import deepcopy
import warnings


def normalize_runtime_mode(value):
    mode = str(value).strip().lower()
    if mode == 'baseline_caravan':
        warnings.warn("'baseline_caravan' is deprecated; use 'caravan' instead.",
                      FutureWarning, stacklevel=2)
        return 'caravan'
    if mode == 'driftadapt':
        return 'drift_conditioned'
    if mode in {'caravan', 'drift_conditioned', 'siteguard'}:
        return mode
    raise ValueError("mode must be caravan, driftadapt (drift_conditioned), or siteguard")


def create_pipeline(config, config_path):
    """Resolve one canonical architecture without mutating the caller's config."""
    mode = normalize_runtime_mode(
        config.get('architecture', config.get('driftadapt', {}).get('mode', 'caravan')))
    if mode == 'siteguard':
        from .siteguard import SiteGuardPipeline
        return SiteGuardPipeline(config, config_path)
    from .pipeline import DriftAdaptPipeline
    config = deepcopy(config)
    config.setdefault('driftadapt', {})['mode'] = mode
    return DriftAdaptPipeline(config, config_path)


def __getattr__(name):
    # Preserve the existing public import without loading PyTorch for CLI help.
    if name == 'DriftAdaptPipeline':
        from .pipeline import DriftAdaptPipeline
        return DriftAdaptPipeline
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = ['DriftAdaptPipeline', 'create_pipeline', 'normalize_runtime_mode']
