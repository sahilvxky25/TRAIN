"""Quick checks.  Run with:  python -m pytest -q   (or:  python tests/test_robot.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from balancebot import ACT_DIM, OBS_DIM, BalancingBotEnv, LinearPolicy, load_policy


def test_env_shapes_and_termination():
    env = BalancingBotEnv(max_steps=50, noise=False, pushes=False)
    obs = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    obs, r, done, info = env.step(np.zeros(ACT_DIM))
    assert obs.shape == (OBS_DIM,) and 0.0 <= r <= 1.0 and not done
    # a zero-torque robot must eventually fall over
    for _ in range(200):
        obs, r, done, info = env.step(np.zeros(ACT_DIM))
        if done:
            break
    assert done


def test_policy_save_load_roundtrip(tmp_path=None):
    import tempfile
    p = LinearPolicy()
    p.M = np.random.default_rng(0).normal(size=(ACT_DIM, OBS_DIM))
    p.norm.mean = np.arange(OBS_DIM, dtype=float)
    p.norm.fixed_std = np.full(OBS_DIM, 2.0)
    path = Path(tempfile.mkdtemp()) / "p.npz"
    p.save(path)
    q = LinearPolicy(); q.load(path)
    obs = np.ones(OBS_DIM)
    assert np.allclose(p.act(obs), q.act(obs))


def test_trained_policy_balances_and_holds_position():
    pol = load_policy()
    env = BalancingBotEnv(max_steps=1000, randomize=False, pushes=False)
    obs, done = env.reset(seed=3, start_offset=(0.3, 0.0)), False
    while not done:
        obs, _, done, info = env.step(pol.act(obs))
    assert not info["fallen"]
    assert info["pos_err"] < 0.10          # within 10 cm after 20 s


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
