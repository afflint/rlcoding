"""Observation-space wrappers for the microbe environments.

The environments in this course expose raw physical coordinates as their
observations: a position in a Petri dish spanning roughly [0, 12] x [0, 6],
and -- from Lecture 5 onward -- additional internal variables on their own
scales. Linear function approximators in Lectures 3-4 never saw these raw
values, because a feature extractor (tile coding, RBFs) stood between the
environment and the agent and produced its own normalised representation.

Neural-network agents have no such intermediary: the network consumes the
observation directly. Feeding raw coordinates of magnitude ten into a
freshly-initialised network is a well-understood nuisance, unrelated to any
reinforcement-learning phenomenon, and it is cleanly removed once and for
all by rescaling observations to the unit hypercube. ``NormalizeObs`` does
exactly that, using the environment's declared observation bounds.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NormalizeObs(gym.ObservationWrapper):
    """Rescale Box observations to the unit hypercube [0, 1]^d.

    The wrapper reads the wrapped environment's ``observation_space`` -- which
    must be a :class:`gymnasium.spaces.Box` with finite bounds -- and maps
    every observation component from its declared range onto [0, 1] by an
    affine transformation. The bounds are fixed and known in advance, so the
    rescaling is deterministic and stateless: unlike a running-statistics
    normaliser, it introduces no dependence on the data seen so far and
    behaves identically during training and evaluation.

    Parameters
    ----------
    env : gymnasium.Env
        Environment to wrap. Its ``observation_space`` must be a ``Box``
        with all-finite ``low`` and ``high``.

    Raises
    ------
    TypeError
        If the wrapped observation space is not a ``Box``.
    ValueError
        If the observation bounds are not all finite, or if any component
        has zero or negative width (``high <= low``).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        space = env.observation_space
        if not isinstance(space, spaces.Box):
            raise TypeError(
                f"NormalizeObs requires a Box observation space, "
                f"got {type(space).__name__}."
            )

        low = np.asarray(space.low, dtype=np.float64)
        high = np.asarray(space.high, dtype=np.float64)

        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError(
                "NormalizeObs requires finite observation bounds; "
                "got a non-finite low or high."
            )

        span = high - low
        if np.any(span <= 0.0):
            raise ValueError(
                "NormalizeObs requires high > low on every component; "
                "got a zero- or negative-width range."
            )

        self._low = low
        self._span = span

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=low.shape, dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Map a raw observation onto [0, 1]^d."""
        scaled = (np.asarray(observation, dtype=np.float64) - self._low) / self._span
        return scaled.astype(np.float32)