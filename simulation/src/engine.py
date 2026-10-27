"""Shared lifecycle and chronological prediction/adaptation loop."""
import json
import random
import time
import numpy as np
import torch
from .config import METHODS, path_for
from .data import FlowData, ReleasedFeedback, benign_classes
from .drift import TrafficDriftInjector
from .metrics import detection_metrics, write_json
from .methods.caravan import Caravan
from .methods.driftadapt import DriftAdapt
from .methods.siteguard import SiteGuard

METHOD_CLASSES = {'caravan':Caravan, 'driftadapt':DriftAdapt, 'siteguard':SiteGuard}


def seed_runtime(config):
    seed = config['runtime']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config['runtime']['threads'])
    torch.use_deterministic_algorithms(True)


def observations(config, data):
    size = config['stream']['window_size']
    injector = TrafficDriftInjector(config['drift_injection'], data.feature_names)
    for index, window in enumerate(data.windows(2, size), 1):
        features, injection = injector.apply(window.features, index)
        # The raw teacher representation must describe the same intervention.
        raw = (features.cpu()*data.span+data.minimum)*data.std+data.mean if injector.enabled else window.raw_features
        timestamps = data.frames[2][data.timestamp_column] if data.timestamp_column else None
        start = float(timestamps.iloc[(index-1)*size]) if timestamps is not None else (index-1)*size
        end = float(timestamps.iloc[index*size-1]) if timestamps is not None else index*size
        yield {'id':index, 'start':start, 'end':end,
               'features':features, 'raw_features':raw.to(data.device),
               'generated_labels':window.generated_labels, 'injection':injection,
               'sample_ids':range((index-1)*size,index*size)}, window.labels.cpu().tolist()


def evaluate(config, data, method, destination):
    method.prepare(data)
    feedback = ReleasedFeedback(config, len(data.labels[2])//config['stream']['window_size']*config['stream']['window_size'])
    benign = benign_classes(config)
    anomaly_target = getattr(method, 'evaluation_target', None) == 'anomaly'
    expected, predicted = [], []
    anomaly_expected, anomaly_predicted = [], []
    count = 0
    inference_seconds = adaptation_seconds = 0.0
    with (destination/'predictions.jsonl').open('w') as predictions, (destination/'events.jsonl').open('w') as events:
        for observation, labels in observations(config, data):
            injection = observation.pop('injection', {})
            observation['feedback_sample_ids'] = feedback.sample_ids
            started = time.perf_counter()
            before = method.state()['adaptation_count']
            bytes_before = method.state()['model_update_bytes']
            prediction = method.infer(observation)
            if config['runtime']['device'].startswith('cuda'):
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter()-started
            values = prediction.tolist()
            if len(values) != len(labels):
                raise ValueError('Method must return exactly one prediction per decision unit')
            if benign is not None:
                valid = [(label,value) for label,value in zip(labels,values) if label is not None]
                anomaly_expected.extend(int(label not in benign) for label,_ in valid)
                anomaly_predicted.extend(int(value) if anomaly_target else int(value not in benign) for _,value in valid)
            if anomaly_target:
                labels = [int(label not in benign) if label is not None else None for label in labels]
            # Ground truth is attached only after prediction; methods never receive it.
            for offset, (value,label) in enumerate(zip(values,labels)):
                row = {'batch_id':observation['id'], 'offset':offset, 'start':observation['start'],
                       'end':observation['end'], 'prediction':int(value), 'label':label,
                       'unit':method.unit, 'model_version':method.state()['adaptation_count']}
                predictions.write(json.dumps(row, allow_nan=False)+'\n')
                count += 1
                if label is not None:
                    expected.append(label)
                    predicted.append(value)
            started = time.perf_counter()
            observation['feedback'] = feedback.release(observation['id'])
            observation['label_budget'] = feedback.remaining
            event = method.adapt(observation, prediction)
            feedback.used += event.get('teacher_samples', 0)
            if config['runtime']['device'].startswith('cuda'):
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            adaptation_seconds += elapsed
            event.update(injection)
            event.update(batch_id=observation['id'], adaptation_seconds=elapsed,
                         adapted=method.state()['adaptation_count'] > before,
                         model_update_bytes=method.state()['model_update_bytes']-bytes_before)
            events.write(json.dumps(event, allow_nan=False)+'\n')
    if not count:
        raise ValueError('Evaluation partition contains no complete decision windows')
    summary = {'method':method.name, 'unit':method.unit, 'predictions':count, 'metrics':detection_metrics(expected,predicted,method.unit),
               'inference_seconds':inference_seconds, 'adaptation_seconds':adaptation_seconds, **method.state()}
    summary['label_definition'] = 'binary_anomaly' if anomaly_target else 'original_class_id'
    if benign is not None:
        summary['anomaly_metrics'] = detection_metrics(anomaly_expected, anomaly_predicted, 'binary_anomaly')
        summary['anomaly_label_mapping'] = {'benign_classes':benign, 'benign_prediction':0, 'attack_prediction':1}
    summary.update(feedback_budget=feedback.budget, total_label_cost_samples=feedback.used,
                   verified_feedback_released=feedback.released, feedback_unreleased=len(feedback.records)-feedback.cursor)
    summary['dropped_partial_flows'] = len(data.labels[2])-count
    write_json(destination/'summary.json', summary)
    return summary


def run(config, selection, phase):
    if selection != 'all' and selection not in METHODS:
        raise ValueError(f'Unknown simulation method: {selection}')
    methods = list(METHODS) if selection == 'all' else [selection]
    root = path_for(config, config['output']['directory'])
    destination = root/config['dataset']['name']/time.strftime('%Y%m%d-%H%M%S')
    destination = destination.with_name(destination.name+f'-{time.time_ns()%1000000000:09d}')
    destination.mkdir(parents=True)
    report = {'phase':phase, 'dataset':config['dataset']['name'], 'timeline':config['timeline'],
              'methods':{}, 'results_directory':str(destination),
              'comparison_policy':'Compare anomaly_metrics only for identical units, partitions and anomaly mappings; original multiclass metrics are separate.'}
    write_json(destination/'configuration.json', {k:v for k,v in config.items() if k != '_base'})
    flow_data = None
    # Missing representations are reported per method without preventing independent runs.
    for name in methods:
        seed_runtime(config)
        method_started = time.perf_counter()
        output = destination/name
        output.mkdir()
        try:
            if flow_data is None:
                flow_data = FlowData(config)
            data = flow_data
            method = METHOD_CLASSES[name](config, output)
            method.name = name
            prepared = {'status':'prepared', 'unit':method.unit,
                        'partition_sizes': [len(p) for p in data.frames]}
            prepared['partition_roles'] = ['training', 'offline_validation', 'test']
            prepared['preprocessing_id'] = data.preprocessing_id
            prepared['input_schema'] = data.schema_report
            if name == 'siteguard':
                prepared['representation'] = method.representation
                prepared['benign_classes'] = method.benign
                prepared['benign_training_flows'] = int(np.isin(data.labels[0].cpu().numpy(), method.benign).sum())
            prepared['timeline_basis'] = config['timeline']['order']
            prepared['wall_clock_chronology_verified'] = config['timeline']['order'] == 'timestamp'
            write_json(output/'prepared.json', prepared)
            result = prepared
            if phase in ('train','all'):
                result = {'status':'trained', 'training':method.train(data)}
            if phase in ('evaluate','all'):
                seed_runtime(config)
                result = {'status':'evaluated', **evaluate(config,data,method,output)}
            result['total_seconds'] = time.perf_counter()-method_started
            report['methods'][name] = result
        except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
            report['methods'][name] = {'status':'failed' if isinstance(exc, RuntimeError) else 'blocked', 'reason':str(exc)}
        write_json(destination/'summary.json', report)
    report['comparable_flow_methods'] = [name for name,row in report['methods'].items()
                                         if row.get('status') == 'evaluated' and row.get('unit') == 'flow']
    write_json(destination/'summary.json', report)
    return report
