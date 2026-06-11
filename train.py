"""
train.py — Entry point for DQN agent training

Usage
-----
    python train.py
    python train.py --n_episodes 500 --lr 1e-3 --epsilon_end 0.05
    python train.py --n_episodes 300 --checkpoint_dir experiments/results

Arguments
---------
    --n_episodes        Number of training episodes (default: 300)
    --lr                Adam learning rate (default: 1e-3)
    --epsilon_start     Initial exploration rate (default: 1.0)
    --epsilon_end       Final exploration rate (default: 0.05)
    --epsilon_decay_ep  Episodes to decay epsilon over (default: 240, i.e. 80% of 300)
    --batch_size        Replay buffer batch size (default: 64)
    --buffer_capacity   Replay buffer capacity (default: 10000)
    --target_sync_every Sync target network every N episodes (default: 10)
    --gamma             Discount factor (default: 0.99)
    --session_budget    Session budget in minutes (default: 120)
    --n_problems        Number of problems to load from processed dataset (default: 500)
    --checkpoint_dir    Directory to save checkpoints (default: experiments/results)
    --log_every         Log metrics every N episodes (default: 10)
    --seed              Random seed for reproducibility (default: 42)
    --problems_path     Path to processed problems CSV (default: data/processed/problems_500.csv)
    --profiles_path     Path to student profiles JSON (default: data/processed/student_profiles.json)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Make sure project root is on sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.dqn import DQNAgent
from src.agent.replay_buffer import ReplayBuffer
from src.agent.trainer import DQNTrainer
from src.environment.env import TrainingEnv
from src.environment.observation_builder import ObservationBuilder, FULL_OBS_DIM
from src.environment.problem import Problem
from src.environment.student_generator import StudentProfile

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the DQN problem-selection agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_episodes",        type=int,   default=300)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--epsilon_start",     type=float, default=1.0)
    parser.add_argument("--epsilon_end",       type=float, default=0.05)
    parser.add_argument("--epsilon_decay_ep",  type=int,   default=None,
                        help="Defaults to 80%% of n_episodes if not set.")
    parser.add_argument("--batch_size",        type=int,   default=64)
    parser.add_argument("--buffer_capacity",   type=int,   default=10_000)
    parser.add_argument("--target_sync_every", type=int,   default=10)
    parser.add_argument("--gamma",             type=float, default=0.99)
    parser.add_argument("--session_budget",    type=float, default=120.0)
    parser.add_argument("--n_problems",        type=int,   default=500)
    parser.add_argument("--checkpoint_dir",    type=str,   default="experiments/results")
    parser.add_argument("--log_every",         type=int,   default=10)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument(
        "--problems_path", type=str,
        default="data/processed/problems_500.csv",
    )
    parser.add_argument(
        "--profiles_path", type=str,
        default="data/processed/student_profiles.json",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def load_problems(path: str, n: int) -> list[Problem]:
    df = pd.read_csv(path)
    if len(df) > n:
        df = df.head(n)
    problems = Problem.from_dataframe(df)
    logger.info(f"Loaded {len(problems)} problems from {path}")
    return problems


def load_profiles(path: str) -> list[StudentProfile]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    profiles = [StudentProfile(**p) for p in raw]
    logger.info(f"Loaded {len(profiles)} student profiles from {path}")
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # Default epsilon decay to 80% of training episodes
    epsilon_decay_ep = args.epsilon_decay_ep or int(args.n_episodes * 0.8)

    # ── Load data ────────────────────────────────────────────────────────────
    problems = load_problems(args.problems_path, args.n_problems)
    profiles = load_profiles(args.profiles_path)

    # ── Build components ─────────────────────────────────────────────────────
    env = TrainingEnv(
        problems           = problems,
        session_budget_min = args.session_budget,
        random_seed        = args.seed,
        profiles           = profiles,
    )

    agent = DQNAgent(
        obs_dim  = FULL_OBS_DIM,
        n_actions = len(problems),
        lr       = args.lr,
        gamma    = args.gamma,
    )

    buffer = ReplayBuffer(capacity=args.buffer_capacity)

    obs_builder = ObservationBuilder(
        problems           = problems,
        session_budget_min = args.session_budget,
    )

    trainer = DQNTrainer(
        env               = env,
        agent             = agent,
        buffer            = buffer,
        obs_builder       = obs_builder,
        batch_size        = args.batch_size,
        epsilon_start     = args.epsilon_start,
        epsilon_end       = args.epsilon_end,
        epsilon_decay_ep  = epsilon_decay_ep,
        target_sync_every = args.target_sync_every,
        checkpoint_dir    = args.checkpoint_dir,
        log_every         = args.log_every,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info(f"  n_episodes       : {args.n_episodes}")
    logger.info(f"  lr               : {args.lr}")
    logger.info(f"  epsilon          : {args.epsilon_start} -> {args.epsilon_end} over {epsilon_decay_ep} ep.")
    logger.info(f"  batch_size       : {args.batch_size}")
    logger.info(f"  buffer_capacity  : {args.buffer_capacity}")
    logger.info(f"  target_sync_every: {args.target_sync_every}")
    logger.info(f"  gamma            : {args.gamma}")
    logger.info(f"  session_budget   : {args.session_budget} min")
    logger.info(f"  problems         : {len(problems)}")
    logger.info(f"  profiles         : {len(profiles)}")
    logger.info(f"  checkpoint_dir   : {args.checkpoint_dir}")
    logger.info("=" * 65)

    history = trainer.train(n_episodes=args.n_episodes)

    # ── Save training history as CSV ─────────────────────────────────────────
    results_dir = Path(args.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    logger.info(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()