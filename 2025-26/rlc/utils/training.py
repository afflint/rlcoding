"""Generic training and evaluation utilities for tabular and function-
approximation agents in the course.

The functions in this module are deliberately agnostic to the specific RL
algorithm used. They assume only the following minimal agent interface:

    agent.select_action(state, *, greedy=False) -> int
    agent.update(state, action, reward, next_state, terminated) -> None
    agent.end_episode() -> None

This contract is satisfied by the Q-learning agent introduced in Lecture 1,
by the SARSA and Monte Carlo agents introduced in Lecture 2, and by the
function-approximation agents introduced in Lectures 3-5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
import gymnasium as gym


class Agent(Protocol):
    """Structural interface implemented by all agents in the course."""

    def select_action(self, state: int, *, greedy: bool = False) -> int: ...
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> None: ...
    def end_episode(self) -> None: ...


@dataclass
class TrainingHistory:
    """Per-episode quantities recorded during training.

    Attributes
    ----------
    episode_returns : list of float
        Sum of rewards collected in each training episode (behaviour policy).
    episode_lengths : list of int
        Number of environment steps in each training episode.
    epsilons : list of float
        Value of the agent's epsilon parameter at the end of each episode,
        if the agent exposes one; otherwise an empty list.
    eval_episodes : list of int
        Indices of the training episodes after which an evaluation batch
        was performed.
    eval_mean_returns : list of float
        Mean return over the evaluation batch, one entry per eval_episodes.
    eval_std_returns : list of float
        Standard deviation of the return over the evaluation batch.
    """

    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    eval_episodes: list[int] = field(default_factory=list)
    eval_mean_returns: list[float] = field(default_factory=list)
    eval_std_returns: list[float] = field(default_factory=list)


def evaluate(
    agent: Agent,
    env: gym.Env,
    n_episodes: int = 20,
    *,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> tuple[float, float, np.ndarray]:
    """Run a batch of greedy evaluation episodes.

    Parameters
    ----------
    agent : Agent
        Trained or partially-trained agent. Must implement
        ``select_action(state, greedy=True)``.
    env : gymnasium.Env
        Evaluation environment. Reset internally; not modified otherwise.
    n_episodes : int
        Number of evaluation episodes to run.
    max_steps : int, optional
        Hard cap on the number of steps per evaluation episode, to guard
        against pathological cases where the greedy policy never terminates.
        ``None`` (default) means unbounded.
    seed : int, optional
        Seed used for the *first* environment reset of the evaluation batch.
        Subsequent resets advance the same RNG, producing a deterministic
        sequence of evaluation episodes.

    Returns
    -------
    mean_return : float
        Mean of the per-episode returns over the batch.
    std_return : float
        Standard deviation of the per-episode returns over the batch.
    returns : np.ndarray
        Array of shape ``(n_episodes,)`` with the individual returns.
    """
    returns = np.zeros(n_episodes, dtype=np.float64)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed if ep == 0 else None)
        ep_return = 0.0
        steps = 0
        while True:
            action = agent.select_action(obs, greedy=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return = ep_return + reward
            steps += 1
            if terminated or truncated:
                break
            if max_steps is not None and steps >= max_steps:
                break
        returns[ep] = ep_return

    return float(returns.mean()), float(returns.std()), returns


def train(
    agent: Agent,
    env: gym.Env,
    n_episodes: int,
    *,
    max_steps_per_episode: Optional[int] = None,
    eval_every: Optional[int] = None,
    eval_episodes: int = 20,
    eval_env: Optional[gym.Env] = None,
    eval_max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    progress: bool = False,
) -> TrainingHistory:
    """Train ``agent`` on ``env`` for ``n_episodes`` episodes.

    Parameters
    ----------
    agent : Agent
        Agent to train. Modified in place.
    env : gymnasium.Env
        Training environment.
    n_episodes : int
        Number of training episodes.
    max_steps_per_episode : int, optional
        Hard cap on the number of steps per training episode. ``None``
        (default) means unbounded; recommended only when the environment
        already enforces a ``TimeLimit``.
    eval_every : int, optional
        If set, run an evaluation batch every ``eval_every`` episodes
        (in addition to one final batch at the end of training).
    eval_episodes : int
        Number of episodes per evaluation batch.
    eval_env : gymnasium.Env, optional
        Environment used for evaluation. Defaults to ``env``. Passing a
        separate instance is recommended so that the evaluation RNG is
        independent of the training RNG.
    eval_max_steps : int, optional
        Hard cap on the steps per evaluation episode.
    seed : int, optional
        Seed for the training environment's *first* reset.
    progress : bool
        If True and the ``tqdm`` package is available, display a progress
        bar over the training episodes.

    Returns
    -------
    TrainingHistory
        Per-episode training data and evaluation snapshots.
    """
    history = TrainingHistory()
    eval_env = eval_env if eval_env is not None else env

    iterator = range(n_episodes)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="Training", unit="ep")
        except ImportError:
            pass  # tqdm not installed; silently fall back to plain range

    for ep in iterator:
        obs, _ = env.reset(seed=seed if ep == 0 else None)
        ep_return = 0.0
        ep_length = 0

        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated)

            obs = next_obs
            ep_return += reward
            ep_length += 1

            if terminated or truncated:
                break
            if max_steps_per_episode is not None and ep_length >= max_steps_per_episode:
                break

        agent.end_episode()

        history.episode_returns.append(ep_return)
        history.episode_lengths.append(ep_length)
        if hasattr(agent, "epsilon"):
            history.epsilons.append(float(agent.epsilon))

        # Periodic evaluation (and a final one at the last episode).
        is_eval_step = eval_every is not None and (
            (ep + 1) % eval_every == 0 or (ep + 1) == n_episodes
        )
        if is_eval_step:
            mean_ret, std_ret, _ = evaluate(
                agent, eval_env,
                n_episodes=eval_episodes,
                max_steps=eval_max_steps,
            )
            history.eval_episodes.append(ep + 1)
            history.eval_mean_returns.append(mean_ret)
            history.eval_std_returns.append(std_ret)

    return history