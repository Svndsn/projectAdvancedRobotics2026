from abc import ABC
from dataclasses import dataclass
from typing import Callable

from brax.training.agents.ppo import train as ppo


@dataclass(frozen=True)
class Args(ABC):
    """
    Configuration arguments for training and environment setup.
    """

    env_name: str
    """Name of the environment to train on."""

    algo: Callable = ppo.train
    """Algorithm to use for training (e.g., 'ppo.train', 'sac.train')."""

    num_timesteps: int = 20_000_000
    """Total number of timesteps to train the agent."""

    num_evals: int = 5
    """Number of evaluations to perform during training."""

    reward_scaling: float = 0.1
    """Scaling factor for rewards."""

    episode_length: int = 1000
    """Maximum length of an episode in steps."""

    normalize_observations: bool = True
    """Whether to normalize observations during training."""

    action_repeat: int = 1
    """Number of times each action is repeated."""

    unroll_length: int = 10
    """Length of unrolled sequences during training."""

    num_minibatches: int = 24
    """Number of minibatches for policy updates."""

    num_updates_per_batch: int = 8
    """Number of updates per batch of data."""

    discounting: float = 0.97
    """Discount factor for future rewards."""

    learning_rate: float = 3e-4
    """Learning rate for the optimizer."""

    entropy_cost: float = 1e-3
    """Coefficient for the entropy regularization term."""

    num_envs: int = 3072
    """Number of parallel environments to run during training."""

    batch_size: int = 512
    """Batch size used during training."""

    seed: int = 0
    """Random seed for reproducibility."""
