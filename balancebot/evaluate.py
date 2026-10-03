"""Evaluation: randomised robustness test + the scripted shove/move-target demo."""
import numpy as np

from .env import BalancingBotEnv


def run_scenario(policy, steps=1500, seed=0):
    """
    Demo timeline (30 s):  start offset 0.4 m -> settle -> shove at 5 s ->
    target moved to x=+0.5 m at 12 s -> shove backwards at 20 s.
    Returns a log of everything needed for plotting / animation.
    """
    env = BalancingBotEnv(max_steps=steps, randomize=False, noise=True, pushes=False)
    obs = env.reset(seed=seed, start_offset=(-0.4, 0.0), target=(0.0, 0.0))
    log = {k: [] for k in ["t", "pitch", "x", "y", "target_x", "ctrl_l", "ctrl_r",
                           "push", "qpos", "xpos", "xmat"]}
    events = {250: ("push", (6.0, 0, 0)), 600: ("target", (0.5, 0.0)),
              1000: ("push", (-6.0, 0, 0))}
    fell = False
    for i in range(steps):
        if i in events:
            kind, val = events[i]
            env.push(val) if kind == "push" else env.set_target(val)
        obs, _, done, info = env.step(policy.act(obs))
        d = env.data
        log["t"].append(i * env.DT)
        log["pitch"].append(info["pitch"])
        log["x"].append(d.qpos[0]); log["y"].append(d.qpos[1])
        log["target_x"].append(env.target[0])
        log["ctrl_l"].append(d.ctrl[0]); log["ctrl_r"].append(d.ctrl[1])
        log["push"].append(float(env.push_start <= env.t - 1 < env.push_end))
        log["xpos"].append(d.xpos.copy()); log["xmat"].append(d.xmat.copy())
        if info["fallen"]:
            fell = True
            break
    log = {k: np.array(v) for k, v in log.items()}
    log["fell"] = fell
    return log


def test(policy, episodes=50):
    # 1) random starts + noise + domain randomisation + random shoves
    env = BalancingBotEnv(max_steps=1000)
    ok, final_err, max_tilt = 0, [], []
    for i in range(episodes):
        obs, done, last = env.reset(seed=50_000 + i), False, None
        tilts = []
        while not done:
            obs, _, done, last = env.step(policy.act(obs))
            tilts.append(abs(last["pitch"]))
        ok += not last["fallen"]
        final_err.append(last["pos_err"]); max_tilt.append(max(tilts))
    print(f"Random-start test ({episodes} eps, 20 s each, with noise, mass/friction "
          f"randomisation and shoves)")
    print(f"  stayed upright : {ok}/{episodes}")
    print(f"  final distance from target : mean {np.mean(final_err) * 100:.1f} cm, "
          f"max {np.max(final_err) * 100:.1f} cm")
    print(f"  worst tilt     : mean {np.degrees(np.mean(max_tilt)):.1f} deg\n")

    # 2) scripted demo
    log = run_scenario(policy)
    t = log["t"]
    def err_at(lo, hi, tgt):
        m = (t >= lo) & (t < hi)
        return abs(np.mean(log["x"][m]) - tgt) * 100 if m.any() else float("nan")
    print("Scripted demo (30 s)  -> fell:", log["fell"])
    m = (t > 5) & (t < 12)
    excursion = np.max(np.abs(log["x"][m])) * 100 if m.any() else float("nan")
    print(f"  hold  (t=3-5 s)     mean position error at 0.0 m : {err_at(3, 5, 0.0):.1f} cm")
    print(f"  max excursion after 6 N shove at 5 s : {excursion:.1f} cm")
    print(f"  back at target before move (t=11-12 s) error  : {err_at(11, 12, 0.0):.1f} cm")
    print(f"  moved target to +0.5 m, error (t=17-20 s)       : {err_at(17, 20, 0.5):.1f} cm")
    print(f"  after backward shove, error (t=28-30 s)         : {err_at(28, 30, 0.5):.1f} cm")
    if log["fell"]:
        print(f"  (robot fell at t = {t[-1]:.1f} s - later metrics are nan)")
