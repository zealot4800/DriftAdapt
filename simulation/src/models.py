"""Shared in-memory packet encoder, classifiers and training."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import OneClassSVM


def fit_model(model, features, targets, settings, *, loss=None, optimizer=None):
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    loss = loss or nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(features, targets), batch_size=settings['batch_size'], shuffle=True)
    model.train()
    for _ in range(settings['epochs']):
        for values, labels in loader:
            optimizer.zero_grad()
            error = loss(model(values), labels)
            if not torch.isfinite(error):
                raise RuntimeError('Nonfinite training loss')
            error.backward()
            optimizer.step()
    model.eval()
    return optimizer


class MLP(nn.Module):
    def __init__(self, inputs, hidden=(16, 8)):
        super().__init__()
        sizes = [inputs, *hidden, 2]
        layers = []
        for index, (left, right) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(sizes)-2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, values):
        return self.layers(values)

    def run_inference(self, values):
        scores = self(values)
        return scores, scores.argmax(1)


def train_classifier(section, config, data):
    if set(data.labels[0].cpu().tolist()) != {0, 1}:
        raise ValueError('Supervised packet-window training requires both source classes')
    model = MLP(len(data.feature_names), section['hidden']).to(data.device)
    fit_model(model, data.features[0], data.labels[0], config['training'])
    return model


def form_training_set(features, labels):
    classes, counts = torch.unique(labels, return_counts=True)
    if classes.tolist() != [0, 1]:
        return None, None
    count = int(counts.min())
    indices = torch.cat([torch.where(labels == label)[0][torch.randperm(int(size), device=labels.device)[:count]]
                         for label, size in zip(classes, counts)])
    return features[indices], labels[indices]


class OneClassDetector:
    """Linear nu-one-class objective with anomaly-positive predictions."""
    def __init__(self, config):
        self.config = config

    def fit_benign(self, values):
        if len(values) < 2:
            raise ValueError('SiteGuard requires at least two benign source windows')
        model = OneClassSVM(kernel='linear', nu=self.config['nu'],
            tol=self.config['tolerance'], max_iter=self.config['max_iterations']).fit(values)
        if model.fit_status_:
            raise RuntimeError('One-class optimization did not converge')
        scale = self.config['nu']*len(values)
        self.weights = -model.coef_.reshape(-1)/scale
        self.rho = float(model.intercept_[0])/scale

    def predict(self, values):
        return (np.asarray(values) @ self.weights-self.rho >= 0).astype(np.int64)


class SourceScaler:
    def fit(self, values):
        self.minimum = values.min(axis=0)
        self.span = np.ptp(values, axis=0)
        self.span = np.where(self.span > 0, self.span, 1.)

    def transform(self, values):
        return (np.asarray(values)-self.minimum)/self.span


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1, bias=False))

    def forward(self, values):
        return values+self.layers(values)


class PacketEncoder(nn.Module):
    """Shared SiteGuard-style DFT and residual sequence autoencoder."""
    def __init__(self, settings, device):
        super().__init__()
        self.settings, self.device = settings, device
        auto = settings['autoencoder']
        channels, length = auto['channels'], settings['sequence_length']
        self.shape = (channels, 5, length)
        size = channels*5*length
        self.encoder = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1),
            *(ResidualBlock(channels) for _ in range(auto['residual_blocks'])),
            nn.ReLU(), nn.Flatten(), nn.Linear(size, auto['latent_dim']))
        self.expand = nn.Linear(auto['latent_dim'], size)
        self.decoder = nn.Sequential(*(ResidualBlock(channels) for _ in range(auto['residual_blocks'])),
                                     nn.ReLU(), nn.Conv2d(channels, 1, 3, padding=1))
        self.register_buffer('minimum', torch.zeros(1, 5, 1))
        self.register_buffer('maximum', torch.zeros(1, 5, 1))
        self.to(device)

    @staticmethod
    def amplitudes(sequences):
        return np.roll(np.abs(np.fft.fft2(sequences, axes=(-2, -1))), (-1, -1), axis=(-2, -1))

    def transform(self, sequences):
        return self._scale(self.amplitudes(sequences))

    def _scale(self, values):
        lower, upper = (self.settings['dft'][key] for key in ('c_min', 'c_max'))
        minimum, maximum = self.minimum.cpu().numpy(), self.maximum.cpu().numpy()
        scaled = np.divide(values-minimum, maximum-minimum, out=np.zeros_like(values), where=maximum > minimum)
        return np.floor(np.clip(scaled, 0, 1)*(upper-lower)+lower).astype(np.float32)

    def forward(self, values):
        return self.decoder(self.expand(self.encoder(values)).reshape(-1, *self.shape))

    def fit(self, windows):
        benign = [window for window in windows if window['label'] == 0]
        if len(benign) < 2:
            raise ValueError('Shared packet encoding requires at least two benign source windows')
        sequences = np.concatenate([window['sequences'][window['mask']] for window in benign])
        if not len(sequences):
            raise ValueError('Benign source windows need data-phase packets')
        amplitudes = self.amplitudes(sequences)
        self.minimum.copy_(torch.from_numpy(amplitudes.min(axis=(0, 2), keepdims=True)))
        self.maximum.copy_(torch.from_numpy(amplitudes.max(axis=(0, 2), keepdims=True)))
        values = torch.from_numpy(self._scale(amplitudes)[:, None]).to(self.device)
        fit_model(self, values, values, self.settings['autoencoder'], loss=nn.MSELoss())

    @torch.inference_mode()
    def vector(self, window):
        mask = window['mask']
        encoded = np.zeros((len(mask), self.settings['autoencoder']['latent_dim']), dtype=np.float32)
        if mask.any():
            values = torch.from_numpy(self.transform(window['sequences'][mask])[:, None]).to(self.device)
            encoded[mask] = self.encoder(values).cpu().numpy()
        return np.concatenate((window['statistics'], encoded.reshape(-1)))
