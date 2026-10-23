"""Configurable pre-activation residual convolutional compression (MSE)."""
import numpy as np
import torch
from torch import nn
from sklearn.svm import OneClassSVM


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(1, channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.layers(x)


class SequenceAutoencoder(nn.Module):
    def __init__(self, config, sequence_length):
        super().__init__()
        self.config = dict(config)
        self.k = sequence_length
        c = config['channels']
        self.encoder = nn.Sequential(nn.Conv2d(1, c, 3, padding=1),
            *[ResidualBlock(c) for _ in range(config['residual_blocks'])],
            nn.Flatten(), nn.Linear(c * 5 * self.k, config['latent_dim']))
        self.expand = nn.Linear(config['latent_dim'], c * 5 * self.k)
        self.decoder = nn.Sequential(
            *[ResidualBlock(c) for _ in range(config['residual_blocks'])],
            nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(self.expand(z).reshape(-1, self.config['channels'], 5, self.k))

    def fit_source(self, x, seed):
        data = torch.as_tensor(x, dtype=torch.float32).unsqueeze(1)
        if len(data) == 0 or not torch.isfinite(data).all():
            raise ValueError('source spectra must be nonempty and finite')
        loader = torch.utils.data.DataLoader(data, batch_size=self.config['batch_size'],
            shuffle=True, generator=torch.Generator().manual_seed(seed))
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        history = []
        self.train()
        for _ in range(self.config['epochs']):
            total = 0.0
            for batch in loader:
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self(batch), batch)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite reconstruction loss')
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            history.append(total / len(data))
        self.eval()
        return history

    @torch.inference_mode()
    def encode(self, x):
        self.eval()
        result = np.empty((len(x), self.config['latent_dim']), dtype=np.float32)
        batch_size = self.config['batch_size']
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32).unsqueeze(1)
            result[start:start + len(batch)] = self.encoder(batch).numpy()
        return result


class OneClassDetector:
    """Slack-corrected linear nu-one-class objective, with anomaly-positive sign.

    Use 0.5||u||² + sum(xi)/(nu*n) - r, xi>=0, u.x>=r-xi; w=-u, rho=-r.
    This explicitly corrects the malformed printed Eq. (1); see SITEGUARD.md.
    """
    def __init__(self, config):
        self.config = dict(config)
        self.weights = None
        self.rho = None

    def fit_benign(self, vectors):
        x = np.asarray(vectors, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError('at least two finite benign source windows are required')
        model = OneClassSVM(kernel='linear', nu=self.config['nu'],
            tol=self.config['tolerance'], max_iter=self.config['max_iterations']).fit(x)
        if model.fit_status_ != 0:
            raise RuntimeError('one-class optimization did not converge')
        # libsvm uses sum(alpha)=nu*n and alpha<=1. Normalize to the stated
        # primal's sum(alpha)=1, alpha<=1/(nu*n); signs alone would conceal
        # this scale difference until the online hinge-loss updates.
        scale = self.config['nu'] * len(x)
        self.weights = -model.coef_.reshape(-1).copy() / scale
        self.rho = float(model.intercept_[0]) / scale
        return self

    def score(self, vectors):
        if self.weights is None:
            raise RuntimeError('classifier has not been trained')
        x = np.asarray(vectors, dtype=np.float64)
        if x.shape[-1] != len(self.weights) or not np.isfinite(x).all():
            raise ValueError('invalid classifier vector')
        return x @ self.weights - self.rho

    def predict(self, vectors):
        return np.where(self.score(vectors) >= 0, 1, -1)


class SourceScaler:
    """Source-only min/max scaling; no mean centering of the linear OCSVM."""
    def fit(self, x):
        self.minimum = np.min(x, axis=0)
        self.span = np.ptp(x, axis=0)
        self.span = np.where(self.span > 0, self.span, 1.0)
        return self

    def transform(self, x):
        # Do not clip target values: preserve evidence of out-of-range traffic.
        return (np.asarray(x) - self.minimum) / self.span
