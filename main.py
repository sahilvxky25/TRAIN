"""Command-line entry point.

    python main.py train            learn a policy      -> models/balance_policy.npz
    python main.py test             robustness numbers + scripted demo
    python main.py plot             plots + side-view GIF -> results/
    python main.py view             live 3-D MuJoCo viewer (needs a display)
"""
import argparse

from balancebot import TrainConfig, load_policy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["train", "test", "plot", "view"], nargs="?", default="test")
    ap.add_argument("--policy", default=None, help="policy .npz to load (default: models/balance_policy.npz)")
    ap.add_argument("--seed", type=int, default=0, help="training seed")
    ap.add_argument("--episodes", type=int, default=50, help="episodes for the random-start test")
    args = ap.parse_args()

    if args.mode == "train":
        from balancebot.ars import train
        train(TrainConfig(seed=args.seed))
        return

    policy = load_policy(args.policy) if args.policy else load_policy()
    if args.mode == "test":
        from balancebot.evaluate import test
        test(policy, episodes=args.episodes)
    elif args.mode == "plot":
        from balancebot.visualize import plot
        plot(policy)
    elif args.mode == "view":
        from balancebot.viewer import view
        view(policy)


if __name__ == "__main__":
    main()
