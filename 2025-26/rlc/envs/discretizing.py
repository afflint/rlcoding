"""A Gymnasium ObservationWrapper that bins a continuous Box observation
into a single discrete index.

Given an environment whose observation space is a 2D ``Box``, the wrapper
partitions the box into a regular ``n_bins_x x n_bins_y`` grid of cells
and replaces the continuous observation with the integer index of the
cell the observation falls in (row-major flattening, matching the
convention used elsewhere in the course).

Agents written for ``Discrete`` observation spaces — including the
``QLearningAgent`` of Lecture 1 — can be applied to a wrapped continuous
environment without modification.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DiscretizingWrapper(gym.ObservationWrapper):
    """Bin a 2D Box observation into a single discrete index.

    Parameters
    ----------
    env : gymnasium.Env
        The environment to wrap. Must have a 2D ``Box`` observation space.
    n_bins_x, n_bins_y : int
        Number of bins along the x and y axes.

    Attributes
    ----------
    n_bins_x, n_bins_y : int
        The grid dimensions.
    n_states : int
        Total number of discrete observations: ``n_bins_x * n_bins_y``.
    bin_size_x, bin_size_y : float
        Width and height of each bin in the original continuous coordinates.
    """

    def __init__(self, env: gym.Env, n_bins_x: int, n_bins_y: int) -> None:
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(
                f"DiscretizingWrapper requires a Box observation space, "
                f"got {type(env.observation_space).__name__}."
            )
        if env.observation_space.shape != (2,):
            raise ValueError(
                f"DiscretizingWrapper requires a 2D observation, "
                f"got shape {env.observation_space.shape}."
            )
        if n_bins_x < 1 or n_bins_y < 1:
            raise ValueError(
                f"n_bins_x and n_bins_y must be positive, got "
                f"({n_bins_x}, {n_bins_y})."
            )

        self.n_bins_x = int(n_bins_x)
        self.n_bins_y = int(n_bins_y)
        self.n_states = self.n_bins_x * self.n_bins_y

        # Original box bounds.
        self._low = env.observation_space.low.astype(np.float64).copy()
        self._high = env.observation_space.high.astype(np.float64).copy()

        self.bin_size_x = float((self._high[0] - self._low[0]) / self.n_bins_x)
        self.bin_size_y = float((self._high[1] - self._low[1]) / self.n_bins_y)

        # Replace the observation space.
        self.observation_space = spaces.Discrete(self.n_states)

    # ------------------------------------------------------------------
    # ObservationWrapper API: only `observation` needs to be overridden.
    # ------------------------------------------------------------------

    def observation(self, observation: np.ndarray) -> int:
        """Map a continuous observation to its discrete bin index."""
        x, y = float(observation[0]), float(observation[1])
        bx = int(np.clip((x - self._low[0]) / self.bin_size_x, 0, self.n_bins_x - 1))
        by = int(np.clip((y - self._low[1]) / self.bin_size_y, 0, self.n_bins_y - 1))
        # Row-major flattening: index = by * n_bins_x + bx.
        return by * self.n_bins_x + bx

    # ------------------------------------------------------------------
    # Inverse map (useful for plotting; not part of the Gym API).
    # ------------------------------------------------------------------

    def bin_center(self, index: int) -> np.ndarray:
        """Return the (x, y) coordinates of the centre of bin ``index``."""
        if not 0 <= index < self.n_states:
            raise ValueError(f"index {index} out of range [0, {self.n_states}).")
        by, bx = divmod(int(index), self.n_bins_x)
        cx = self._low[0] + (bx + 0.5) * self.bin_size_x
        cy = self._low[1] + (by + 0.5) * self.bin_size_y
        return np.array([cx, cy], dtype=np.float64)