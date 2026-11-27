"""Configuration and explicit experiment boundaries."""
from pathlib import Path
import yaml

METHODS = ('caravan', 'siteguard', 'driftadapt')


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a YAML mapping')
    config['_base'] = path.parent
    for key in ('dataset', 'timeline', 'features', 'model', 'labeler', 'stream', 'training', 'trigger', 'siteguard', 'runtime'):
        if key not in config:
            raise ValueError(f'Missing configuration section: {key}')
    if config['runtime']['device'].startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            raise ValueError('Configured CUDA device is unavailable; select cpu in YAML')
    if type(config['stream']['buffer_windows']) is not int or config['stream']['buffer_windows'] < 2:
        raise ValueError('stream.buffer_windows must be an integer of at least 2')
    if config['timeline'].get('order') != 'timestamp':
        raise ValueError('timeline.order must be timestamp for packet simulations')
    for section in (config['training'], config['features']['autoencoder']):
        if (any(type(section[key]) is not int or section[key] < 1 for key in ('epochs', 'batch_size'))
                or not 0 < section['learning_rate'] < float('inf')):
            raise ValueError('Training epochs, batch size and learning rate must be positive and finite')
    auto = config['features']['autoencoder']
    if any(type(auto[key]) is not int or auto[key] < 1 for key in ('channels', 'residual_blocks', 'latent_dim')):
        raise ValueError('Autoencoder dimensions and residual blocks must be positive integers')
    dft = config['features']['dft']
    if not float('-inf') < dft['c_min'] < dft['c_max'] < float('inf'):
        raise ValueError('Finite dft.c_min must be less than c_max')
    return config


def path_for(config, value):
    if not value:
        raise ValueError('Required input path is not configured')
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config['_base'] / path).resolve()
