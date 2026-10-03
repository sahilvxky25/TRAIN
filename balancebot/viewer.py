"""Live interactive 3-D MuJoCo viewer (needs a display)."""
import time

from .env import BalancingBotEnv


def view(policy):
    import mujoco.viewer
    env = BalancingBotEnv(max_steps=10 ** 9, randomize=False, pushes=False)
    obs = env.reset(seed=1, start_offset=(-0.4, 0.0))
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        # Tip: in the viewer, double-click the robot then Ctrl+right-drag to shove it.
        while v.is_running():
            t0 = time.time()
            obs, _, done, info = env.step(policy.act(obs))
            env.model.site_pos[env.target_site][:2] = env.target
            v.sync()
            if done and info["fallen"]:
                obs = env.reset(start_offset=(-0.4, 0.0))
            time.sleep(max(0.0, env.DT - (time.time() - t0)))
