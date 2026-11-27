"""SiteGuard's benign-only linear detector on the shared packet encoding."""
from ..models import OneClassDetector, SourceScaler


class SiteGuard:
    unit = 'detection_window'

    def __init__(self, config):
        self.detector = OneClassDetector(config['siteguard']['classifier'])
        self.scaler = SourceScaler()

    def train(self, data):
        values = data.features[0][data.labels[0] == 0].cpu().numpy()
        self.scaler.fit(values)
        self.detector.fit_benign(self.scaler.transform(values))

    def infer(self, observation):
        values = observation['features'].detach().cpu().numpy()
        return self.detector.predict(self.scaler.transform(values))

    def adapt(self, observation, prediction):
        pass

    def state(self):
        return {'adaptation_count': 0, 'labeling_cost_samples': 0}
