"""Configuration and explicit experiment boundaries."""
from pathlib import Path
import yaml

METHODS = ('caravan', 'driftadapt', 'siteguard')
PHASES = ('prepare', 'train', 'evaluate', 'all')


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a YAML mapping')
    config['_base'] = path.parent
    config['_path'] = str(path)
    for key in ('dataset', 'timeline', 'runtime', 'output'):
        if key not in config:
            raise ValueError(f'Missing configuration section: {key}')
    if config['runtime']['device'].startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            raise ValueError('Configured CUDA device is unavailable; select cpu in YAML')
    if config['stream']['window_size'] < 1:
        raise ValueError('stream.window_size must be positive')
    if config['timeline'].get('order') not in ('timestamp', 'provided'):
        raise ValueError('timeline.order must explicitly be timestamp or provided')
    return config


def path_for(config, value):
    if not value:
        raise ValueError('Required input/checkpoint path is not configured')
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config['_base'] / path).resolve()
