"""MuJoCo environment: a two-wheeled robot that must balance and hold a position."""
from pathlib import Path

import numpy as np
import mujoco

ASSET_PATH = Path(__file__).parent / "assets" / "robot.xml"


def load_model():
    """Load the robot model (x = forward, y = axle direction, z = up)."""
    return mujoco.MjModel.from_xml_path(str(ASSET_PATH))


class BalancingBotEnv:
    """MuJoCo environment. Control runs at 50 Hz (10 physics steps of 2 ms)."""

    FRAME_SKIP = 10
    DT = 0.002 * FRAME_SKIP
    MAX_TILT = 0.6                      # rad (~34 deg) -> counts as fallen

    def __init__(self, max_steps=500, randomize=True, noise=True, pushes=True):
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.randomize, self.noise, self.pushes = randomize, noise, pushes

        m = self.model
        self.cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.target_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target")
        self._mass0 = m.body_mass.copy()
        self._inertia0 = m.body_inertia.copy()
        self._friction0 = m.geom_friction.copy()

        self.pos_w = (2.0, 5.0)         # (linear, quadratic) position-error cost weights
        self.target = np.zeros(2)       # world (x, y) position to hold
        self.e_int = 0.0                # leaky integral of the forward position error
        self.rng = np.random.default_rng()
        self.t = 0
        self.push_force = np.zeros(3)
        self.push_start = self.push_end = -1

    # ---- helpers -----------------------------------------------------------
    def _pose(self):
        """pitch (+ = leaning forward), roll, yaw of the chassis."""
        R = self.data.xmat[self.cid].reshape(3, 3)
        pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return pitch, roll, yaw

    def _obs(self):
        d = self.data
        pitch, roll, yaw = self._pose()
        w = d.qvel[3:6]                                  # body-frame angular vel
        err = self.target - d.qpos[0:2]                  # world position error
        v = d.qvel[0:2]
        c, s = np.cos(yaw), np.sin(yaw)
        e_f, e_l = c * err[0] + s * err[1], -s * err[0] + c * err[1]
        v_f, v_l = c * v[0] + s * v[1], -s * v[0] + c * v[1]
        self.e_int = float(np.clip(0.995 * self.e_int + e_f * self.DT, -0.5, 0.5))
        obs = np.array([pitch, w[1], roll, e_f, v_f, yaw, w[2], e_l, v_l, self.e_int])
        if self.noise:                                   # imperfect IMU / encoders
            obs[:3] += self.rng.normal(0, [0.004, 0.03, 0.004])
            obs[3:9] += self.rng.normal(0, [0.002, 0.01, 0.004, 0.03, 0.002, 0.01])
        return obs

    # ---- gym-style API -----------------------------------------------------
    def reset(self, seed=None, start_offset=None, target=(0.0, 0.0)):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d, r = self.model, self.data, self.rng

        # domain randomisation: mass +-15 %, wheel friction 0.7-1.3
        m.body_mass[:] = self._mass0
        m.body_inertia[:] = self._inertia0
        m.geom_friction[:] = self._friction0
        if self.randomize:
            k = r.uniform(0.85, 1.15)
            m.body_mass[self.cid] *= k
            m.body_inertia[self.cid] *= k
            m.geom_friction[:, 0] *= r.uniform(0.7, 1.3)

        mujoco.mj_resetData(m, d)
        pitch0 = r.uniform(-0.08, 0.08)
        yaw0 = r.uniform(-0.2, 0.2)
        if start_offset is None:
            start_offset = (r.uniform(-0.6, 0.6), r.uniform(-0.1, 0.1))
        d.qpos[0:2] = np.array(target) + np.array(start_offset)
        # orientation: yaw about z, then pitch about y
        qz = np.array([np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2)])
        qy = np.array([np.cos(pitch0 / 2), 0, np.sin(pitch0 / 2), 0])
        mujoco.mju_mulQuat(d.qpos[3:7], qz, qy)
        mujoco.mj_forward(m, d)

        self.target = np.array(target, dtype=float)
        self.t = 0
        self.e_int = 0.0
        self.push_start = self.push_end = -1
        if self.pushes and r.random() < 0.7:           # one random shove / episode
            self.push_start = int(r.integers(50, max(60, self.max_steps - 120)))
            self.push_end = self.push_start + 5          # 0.1 s pulse
            self.push_force = np.array([r.uniform(-4, 4), r.uniform(-1.5, 1.5), 0.0])
        return self._obs()

    def push(self, force_xyz, duration_steps=5):
        """Manually shove the robot (used in the test / demo)."""
        self.push_force = np.array(force_xyz, dtype=float)
        self.push_start, self.push_end = self.t, self.t + duration_steps

    def set_target(self, xy):
        self.target = np.array(xy, dtype=float)

    def step(self, action):
        d = self.data
        a = np.clip(action, -1, 1)
        drive, turn = a
        d.ctrl[0] = np.clip(drive - turn, -1, 1)         # left wheel
        d.ctrl[1] = np.clip(drive + turn, -1, 1)         # right wheel

        d.xfrc_applied[self.cid, :3] = (
            self.push_force if self.push_start <= self.t < self.push_end else 0.0)
        mujoco.mj_step(self.model, d, nstep=self.FRAME_SKIP)
        self.t += 1

        obs = self._obs()
        pitch, roll, yaw = self._pose()
        e = self.target - d.qpos[0:2]
        dist = np.linalg.norm(e)
        # the linear + absolute position terms make even a few cm of error cost
        # something, which is what teaches the robot to hold its spot precisely
        cost = (3.0 * pitch ** 2 + self.pos_w[0] * dist + self.pos_w[1] * dist ** 2
                + 0.3 * (d.qvel[0:2] @ d.qvel[0:2])
                + 0.5 * yaw ** 2 + 0.05 * d.qvel[5] ** 2 + 0.02 * (a @ a))
        fallen = abs(pitch) > self.MAX_TILT or abs(roll) > self.MAX_TILT
        done = fallen or self.t >= self.max_steps
        # always-positive reward in (0, 1]: staying alive is never worse than falling
        reward = 0.0 if fallen else float(np.exp(-cost))
        return obs, reward, done, {"fallen": fallen, "pitch": pitch,
                                   "pos_err": float(np.linalg.norm(e))}
