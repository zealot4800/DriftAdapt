"""Train and evaluate the three methods in memory on one packet pipeline."""
import random
import time
from types import SimpleNamespace
import numpy as np
import torch
from .config import METHODS
from .packets import PacketData
from .observations import ReferenceChannel
from .metrics import detection_metrics
from .methods.caravan import Caravan
from .methods.siteguard import SiteGuard
from .methods.driftadapt import DriftAdapt

METHOD_CLASSES = {'caravan': Caravan, 'siteguard': SiteGuard, 'driftadapt': DriftAdapt}


def seed_runtime(config):
    seed = config['runtime']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config['runtime']['threads'])
    torch.use_deterministic_algorithms(True)


def evaluate(data, method, config):
    channel = ReferenceChannel(config.get('observation', {}), config['runtime']['seed'])
    if isinstance(method, DriftAdapt):
        method.bind_reference(channel.request, channel.advance, channel.stats)
    expected, predicted = [], []
    for original, labels in data.observations():
        canonical = original['features'].detach().cpu().numpy()[0]
        observed = channel.observe(original['id'], original['start'], canonical)
        values = torch.as_tensor(observed, dtype=torch.float32, device=data.device)[None]
        observation = {key: original[key] for key in ('id', 'start', 'end')}
        observation['features'] = values
        if not isinstance(method, DriftAdapt):
            observation['raw_features'] = original['raw_features']+(values-original['features'])*data.scale
        prediction = method.infer(observation)
        values = prediction.tolist()
        if len(values) != 1 or values[0] not in (0, 1):
            raise ValueError('Method must return one binary prediction per packet window')
        if labels[0] is not None:
            expected.append(labels[0])
            predicted.append(int(values[0]))
        method.adapt(observation, prediction)
    return {'windows': len(data.partitions[2]), 'metrics': detection_metrics(expected, predicted), **method.state()}


def run(config, selection='all'):
    if selection not in (*METHODS, 'all'):
        raise ValueError('Unknown simulation method')
    seed_runtime(config)
    data = PacketData(config)
    data.prepare()
    training = SimpleNamespace(features=data.features[:2], labels=data.labels[:2],
                               feature_names=data.feature_names, device=data.device)
    results = {}
    for name in METHODS if selection == 'all' else (selection,):
        started = time.perf_counter()
        try:
            seed_runtime(config)
            method_config = ({'training': config['training'], 'runtime': config['runtime'],
                              'driftadapt': config.get('driftadapt', {})} if name == 'driftadapt' else config)
            method = METHOD_CLASSES[name](method_config)
            method.train(training)
            seed_runtime(config)
            results[name] = evaluate(data, method, config)
            if config['runtime']['device'].startswith('cuda'):
                torch.cuda.synchronize()
            results[name]['seconds'] = time.perf_counter()-started
        except (ValueError, RuntimeError) as exc:
            results[name] = {'error': str(exc)}
    return results
