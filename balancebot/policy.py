"""Linear policy + running observation normaliser (the thing ARS trains)."""
from pathlib import Path

import numpy as np

from .config import ACT_DIM, OBS_DIM, POLICY_PATH


class RunningNorm:
    def __init__(self, n):
        self.n, self.mean, self.m2 = 0, np.zeros(n), np.zeros(n)
        self.fixed_std = None                   # set when a trained policy is loaded

    def update(self, batch):                    # batch: (k, n)
        for x in batch:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (x - self.mean)

    @property
    def std(self):
        if self.fixed_std is not None:
            return self.fixed_std
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(np.maximum(self.m2 / self.n, 1e-4))


class LinearPolicy:
    def __init__(self):
        self.M = np.zeros((ACT_DIM, OBS_DIM))
        self.norm = RunningNorm(OBS_DIM)

    def act(self, obs, M=None):
        M = self.M if M is None else M
        return np.clip(M @ ((obs - self.norm.mean) / self.norm.std), -1, 1)

    def save(self, path=POLICY_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, M=self.M, mean=self.norm.mean, std=self.norm.std)

    def load(self, path=POLICY_PATH):
        z = np.load(path)
        self.M = z["M"]
        self.norm.mean = z["mean"]
        self.norm.fixed_std = z["std"]


def load_policy(path=POLICY_PATH):
    p = LinearPolicy()
    p.load(path)
    return p
