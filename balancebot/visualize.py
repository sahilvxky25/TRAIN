"""Headless plots and a side-view GIF drawn from MuJoCo's own state."""
import mujoco
import numpy as np

from .config import RESULTS_DIR
from .env import load_model
from .evaluate import run_scenario


def plot(policy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    png, gif = RESULTS_DIR / "mujoco_balance_plots.png", RESULTS_DIR / "mujoco_balance.gif"
    log = run_scenario(policy)
    t = log["t"]

    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(t, np.degrees(log["pitch"]), color="tab:blue")
    ax[0].set_ylabel("pitch (deg)"); ax[0].axhline(0, color="k", lw=.5)
    ax[1].plot(t, log["x"], label="robot x"); ax[1].plot(t, log["target_x"], "g--", label="target x")
    ax[1].set_ylabel("position (m)"); ax[1].legend(loc="upper left")
    ax[2].plot(t, log["ctrl_l"], label="left motor"); ax[2].plot(t, log["ctrl_r"], label="right motor", alpha=.7)
    ax[2].set_ylabel("motor command"); ax[2].set_xlabel("time (s)"); ax[2].legend(loc="upper right")
    for a in ax:
        for ts in t[np.flatnonzero(np.diff(log["push"]) > 0)]:
            a.axvline(ts, color="red", alpha=.5, ls=":")
    ax[0].set_title("RL balancing robot in MuJoCo (red dotted = shove)")
    fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
    print(f"Saved {png}")

    # ---- side-view animation -------------------------------------------
    m = load_model()
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    wl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_l")
    box = np.array([[-.04, .02], [.04, .02], [.04, .26], [-.04, .26]])    # (x, z) in body frame

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(-1.0, 1.0); ax.set_ylim(-0.05, 0.6); ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=1.5)
    tgt, = ax.plot([], [], "g^", ms=12)
    body, = ax.plot([], [], "-", lw=3, color="tab:blue")
    wheel, = ax.plot([], [], "o", ms=28, color="k")
    head, = ax.plot([], [], "o", ms=10, color="tab:red")
    arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(color="red", lw=3))
    txt = ax.text(-0.95, 0.52, "")
    idx = range(0, len(t), 2)

    def draw(k):
        i = idx[k]
        R = log["xmat"][i][cid].reshape(3, 3); p = log["xpos"][i][cid]
        pts = np.array([p[[0, 2]] + R[np.ix_([0, 2], [0, 2])] @ q for q in box])
        body.set_data(np.r_[pts[:, 0], pts[0, 0]], np.r_[pts[:, 1], pts[0, 1]])
        w = log["xpos"][i][wl]
        wheel.set_data([w[0]], [w[2]])
        top = p[[0, 2]] + R[np.ix_([0, 2], [0, 2])] @ np.array([0, .28])
        head.set_data([top[0]], [top[1]])
        tgt.set_data([log["target_x"][i]], [0.0])
        if log["push"][i]:
            direction = 1 if (i < 800) else -1
            arrow.xy, arrow.xytext = (p[0], .3), (p[0] - .25 * direction, .3)
            arrow.set_visible(True)
        else:
            arrow.set_visible(False)
        ax.set_xlim(p[0] - 1.0, p[0] + 1.0)
        txt.set_position((p[0] - .95, .52))
        txt.set_text(f"t={t[i]:4.1f}s  pitch={np.degrees(log['pitch'][i]):+5.1f}deg  "
                     f"x={log['x'][i]:+.2f} (target {log['target_x'][i]:+.2f})")
        return body, wheel, head, tgt, arrow, txt

    FuncAnimation(fig, draw, frames=len(idx), blit=False).save(
        gif, writer=PillowWriter(fps=25))
    plt.close(fig)
    print(f"Saved {gif}")
