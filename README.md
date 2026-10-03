# RL Self-Balancing, Position-Holding Robot (MuJoCo)

A two-wheeled robot simulated in MuJoCo learns, by reinforcement learning, to
**stay upright**, **hold a position**, recover from shoves, and move to a new
target position and hold it there.

* **Algorithm:** Augmented Random Search (ARS-V2), a model-free RL method, training a linear policy (numpy only).
* **Observation (10):** pitch, pitch rate, roll, forward error, forward velocity, yaw, yaw rate, lateral error, lateral velocity, integral of forward error.
* **Action (2):** `[drive, turn]`, mixed into left/right wheel motor commands.
* **Reward per step:** `exp(-cost)`; cost penalises tilt, distance from the target, speed, heading error and control effort. Falling ends the episode with reward 0.
* **Robustness during training:** sensor noise, random mass (+-15 %) and wheel friction (0.7-1.3x), random shoves.

## Layout

```
mujoco_balancing_robot/
├── main.py                  CLI: train | test | plot | view
├── requirements.txt
├── balancebot/
│   ├── config.py            dimensions, paths, TrainConfig (hyper-parameters)
│   ├── assets/robot.xml     MuJoCo model (MJCF) - edit to change the robot
│   ├── env.py               BalancingBotEnv: physics, observations, reward, shoves
│   ├── policy.py            LinearPolicy, RunningNorm, load_policy
│   ├── ars.py               rollout() + train() (curriculum ARS)
│   ├── evaluate.py          run_scenario() demo, test() robustness numbers
│   ├── visualize.py         plots + side-view GIF (headless)
│   └── viewer.py            live 3-D viewer (needs a display)
├── models/balance_policy.npz   trained policy
├── results/                    generated plots / GIF
└── tests/test_robot.py
```

## Usage

```bash
pip install -r requirements.txt
python main.py test      # uses models/balance_policy.npz
python main.py plot      # writes results/mujoco_balance_plots.png and .gif
python main.py view      # live MuJoCo window (double-click robot, Ctrl+right-drag to shove)
python main.py train     # ~8 min on one CPU core; overwrites models/balance_policy.npz
python -m pytest -q      # or: python tests/test_robot.py
```

## Results of the included policy

| Test | Result |
|---|---|
| 50 random 20 s runs (noise, randomisation, shoves) | 50/50 upright, final distance mean 8.8 cm, max 21.8 cm |
| 6 N shove while holding | peak excursion 13.7 cm, returns to target |
| Target moved to +0.5 m | 0.1 cm error after settling |
| Backward shove | 0.0 cm error after settling |

## Known limitations

* Motor commands are jittery (they chatter and often saturate) because the policy reacts strongly to noisy sensors. Adding a control-smoothness penalty in `env.py` or filtering the sensors is the obvious next step.
* Position response is slow and overshoots (about 0.65 m for a 0.5 m move).
* The integral term covers the forward axis only; sideways position is held indirectly by holding heading.
* `view` needs a display and was not run in the headless development environment.

## Where to change things

| I want to... | Edit |
|---|---|
| change robot size / mass / motor strength | `balancebot/assets/robot.xml` |
| change the reward or add observations | `balancebot/env.py` (and `OBS_DIM` in `config.py`) |
| change learning hyper-parameters / curriculum | `TrainConfig` in `balancebot/config.py` |
| change the demo scenario | `run_scenario()` in `balancebot/evaluate.py` |

If you change the observation size or the robot, retrain; the saved policy no longer matches.
