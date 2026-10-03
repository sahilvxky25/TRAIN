"""Augmented Random Search (ARS-V2): model-free RL for a linear policy."""
import time

import numpy as np

from .config import ACT_DIM, OBS_DIM, POLICY_PATH, TrainConfig
from .env import BalancingBotEnv
from .policy import LinearPolicy


def rollout(env, policy, M, seed, collect=False):
    obs = env.reset(seed=seed)
    total, observations, done = 0.0, [], False
    while not done:
        if collect:
            observations.append(obs)
        obs, r, done, _ = env.step(policy.act(obs, M))
        total += r
    return total, observations


def train(cfg: TrainConfig = TrainConfig()):
    """
    Curriculum: stage 1 learns to BALANCE (gentle position penalty), stage 2 keeps
    the learned policy and sharpens POSITION HOLDING (strict position penalty).
    Policies are always compared with the strict (final) reward.
    """
    stages, n_dirs, step_size, noise_std, seed = (
        cfg.stages, cfg.n_dirs, cfg.step_size, cfg.noise_std, cfg.seed)
    rng = np.random.default_rng(seed)
    env = BalancingBotEnv(max_steps=500)
    eval_env = BalancingBotEnv(max_steps=1000)            # strict weights (default)
    policy = LinearPolicy()
    best_score, best = -np.inf, None
    t0, it = time.time(), 0

    for n_iters, weights in stages:
        env.pos_w = weights
        print(f"--- stage: {n_iters} iterations, position cost weights {weights}", flush=True)
        for _ in range(n_iters):
            it += 1
            deltas = rng.standard_normal((n_dirs, ACT_DIM, OBS_DIM))
            r_plus, r_minus, seen = np.zeros(n_dirs), np.zeros(n_dirs), []

            for k in range(n_dirs):
                ep_seed = int(rng.integers(1 << 30))   # same start for +/- (fair test)
                r_plus[k], o1 = rollout(env, policy, policy.M + noise_std * deltas[k], ep_seed, True)
                r_minus[k], o2 = rollout(env, policy, policy.M - noise_std * deltas[k], ep_seed, True)
                seen += o1 + o2

            # ARS update: move along directions whose +/- rollouts differ the most
            order = np.argsort(-np.maximum(r_plus, r_minus))[: max(1, n_dirs // 2)]
            sigma_r = np.std(np.concatenate([r_plus[order], r_minus[order]])) + 1e-8
            grad = sum((r_plus[k] - r_minus[k]) * deltas[k] for k in order)
            policy.M += step_size / (len(order) * sigma_r) * grad
            policy.norm.update(np.array(seen))

            if it % 10 == 0:
                scores = [rollout(eval_env, policy, policy.M, 10_000 + i)[0] for i in range(5)]
                score = float(np.mean(scores))
                if score > best_score:
                    best_score = score
                    best = (policy.M.copy(), policy.norm.mean.copy(), policy.norm.std.copy())
                print(f"iter {it:4d} | eval return {score:7.1f} (max 1000, strict reward)"
                      f" | best {best_score:7.1f} | {time.time() - t0:5.0f}s", flush=True)

    policy.M = best[0]
    policy.norm.mean = best[1]
    policy.norm.fixed_std = best[2]
    policy.save()
    print(f"Saved {POLICY_PATH}  (now run:  python main.py test)")
    return policy
