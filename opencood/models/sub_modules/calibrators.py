# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm


def softplus(x):
    """Transform the input to positive output."""
    return F.softplus(x, 1.0, 20.0) + 1e-6


def inv_softplus(y):
    """Inverse softplus function."""
    if torch.any(y <= 0.0):
        raise ValueError("Input to `inv_softplus` must be positive.")
    _y = y - 1e-6
    return _y + torch.log(-torch.expm1(-_y))


def BCE(yhat, y):
    """Compute binary cross-entropy loss for a vector of predictions

    Parameters
    ----------
    yhat
        An array with len(yhat) predictions between [0, 1]
    y
        An array with len(y) labels where each is one of {0, 1}
    """
    yhat = np.clip(yhat, 1e-6, 1 - 1e-6)
    return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)).mean()


class Calibrator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-2

    def preprocess_probs(self, probs):
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        probs = probs.view(-1, 1).float()
        probs = torch.clip(probs, self.epsilon, 1 - self.epsilon)
        return probs

    def preprocess_labels(self, labels):
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        labels = labels.view(-1, 1).float()
        return labels

    def fit(self, probs, labels, num_iter=300):
        probs = self.preprocess_probs(probs)
        labels = self.preprocess_labels(labels)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.LBFGS(self.parameters())

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            new_probs = self.forward(probs)
            loss = criterion(new_probs, labels)
            if loss.requires_grad:
                loss.backward()
            return loss

        self.train()
        old_loss = None
        for _ in tqdm(range(num_iter)):
            loss = optimizer.step(closure)
            if old_loss is None:
                old_loss = loss.item()
            else:
                delta = abs(loss - old_loss)
                if delta < 1e-2:
                    break
        self.eval()

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def transform(self, probs):
        raise NotImplementedError

    def forward(self, probs):
        raise NotImplementedError


class DoublyBoundedScaling(Calibrator):

    def __init__(self):
        super().__init__()
        self._a = Parameter(inv_softplus(torch.tensor(1.0)))
        self._b = Parameter(inv_softplus(torch.tensor(1.0)))

    @property
    def a(self):
        return softplus(self._a)

    @property
    def b(self):
        return softplus(self._b)

    def forward(self, probs):
        # inverse sigmoid
        probs = torch.sigmoid(probs)
        new_probs = 1 - (1 - probs**self.a)**self.b * (1-2e-6) + 1e-6
        new_probs = torch.logit(new_probs)
        return new_probs

    def transform(self, probs):
        with torch.no_grad():
            probs = self.preprocess_probs(probs)
            new_probs = self(probs)
        return new_probs.numpy().ravel()


class PlattScaling(Calibrator):

    def __init__(self):
        super().__init__()
        self.w = Parameter(torch.tensor(1.0))
        self.b = Parameter(torch.tensor(0.0))

    def forward(self, probs):
        new_probs = self.w * probs + self.b
        return new_probs

    def transform(self, probs):
        with torch.no_grad():
            probs = self.preprocess_probs(probs)
            new_probs = self(probs)
        return new_probs.numpy().ravel()


class TemperatureScaling(PlattScaling):

    def __init__(self):
        super().__init__()
        self.b.requires_grad = False