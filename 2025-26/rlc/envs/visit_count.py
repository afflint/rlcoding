"""A Gymnasium wrapper that records state-action visit counts.

The wrapper maintains an integer matrix of shape (n_states, n_actions),
incremented every time the agent commands an action in a given state.
It is intended for environments with discrete state and action spaces;
applying it to environments with continuous spaces will raise an error
at construction time.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class VisitCountWrapper(gym.Wrapper):
    """Track state-action visit counts of the wrapped environment.

    The wrapper stores a matrix ``state_action_counts`` of shape
    ``(observation_space.n, action_space.n)``, where entry ``[s, a]``
    is the number of times the agent commanded action ``a`` while in
    state ``s``. The aggregated marginal ``state_counts[s] = sum_a N[s, a]``
    is exposed as a property.

    Parameters
    ----------
    env : gymnasium.Env
        The environment to wrap. Must have ``Discrete`` observation and
        action spaces.

    Attributes
    ----------
    state_action_counts : np.ndarray of int64, shape (n_states, n_actions)
        Cumulative count of (state, action) pairs across all episodes
        seen since construction (or since the last call to ``reset_counts``).

    Notes
    -----
    Counts are *cumulative across episodes*: they are not reset by
    ``reset()``. If you want to start counting afresh, call
    ``reset_counts()`` explicitly. This separation of concerns mirrors
    the design of ``RecordEpisodeStatistics``, which similarly keeps
    cross-episode statistics across resets.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Discrete):
            raise TypeError(
                f"VisitCountWrapper requires a Discrete observation space, "
                f"got {type(env.observation_space).__name__}."
            )
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError(
                f"VisitCountWrapper requires a Discrete action space, "
                f"got {type(env.action_space).__name__}."
            )

        self.n_states: int = int(env.observation_space.n)
        self.n_actions: int = int(env.action_space.n)
        self.state_action_counts: np.ndarray = np.zeros(
            (self.n_states, self.n_actions), dtype=np.int64
        )
        # Tracks the state we are currently in, between consecutive step calls.
        self._current_state: Optional[int] = None

    # ------------------------------------------------------------------
    # Wrapped Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._current_state = int(obs)
        return obs, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self._current_state is None:
            raise RuntimeError("step() called before reset() on the wrapper.")
        # Record the (s, a) pair *before* the transition.
        self.state_action_counts[self._current_state, int(action)] += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Update the tracked state to the new observation.
        self._current_state = int(obs)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Public utilities
    # ------------------------------------------------------------------

    @property
    def state_counts(self) -> np.ndarray:
        """Marginal state visit counts: N(s) = sum_a N(s, a)."""
        return self.state_action_counts.sum(axis=1)

    def reset_counts(self) -> None:
        """Zero the cumulative visit-count matrix."""
        self.state_action_counts.fill(0)