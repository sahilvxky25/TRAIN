"""Central configuration: dimensions, file locations and training hyper-parameters."""
from dataclasses import dataclass
from pathlib import Path

# Policy interface ------------------------------------------------------------
# Observation (10): pitch, pitch rate, roll, forward error, forward velocity,
#                   yaw, yaw rate, lateral error, lateral velocity,
#                   integral of forward error (removes steady offsets)
# Action (2): [drive, turn] -> left / right wheel motor command
OBS_DIM, ACT_DIM = 10, 2

# Locations (relative to the project root) ------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = PROJECT_ROOT / "models" / "balance_policy.npz"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class TrainConfig:
    """ARS hyper-parameters.

    `stages` is a curriculum of (iterations, (linear_w, quadratic_w)) where the
    weights set how strongly position error is penalised. Stage 1 teaches the
    robot to balance (gentle penalty), stage 2 sharpens position holding.
    """
    stages: tuple = ((40, (0.0, 3.0)), (140, (2.0, 5.0)))
    n_dirs: int = 16          # random search directions per iteration
    step_size: float = 0.1    # learning rate
    noise_std: float = 0.2    # exploration noise on the policy matrix
    seed: int = 0
