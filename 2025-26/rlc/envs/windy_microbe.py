"""Windy variant of the continuous microbe environment.

A drop-in subclass of ``ContinuousMicrobeEnv`` that adds a uniform velocity
field — a constant drift — to the displacement at every step. With default
arguments (``current=(0, 0)``) the subclass is operationally identical to
the parent class.

The override is intentionally minimal: only ``_intended_landing`` is touched.
Reward, geometry, termination, rendering, and the toxic-teleport mechanism
are inherited unchanged.
"""

from __future__ import annotations

import numpy as np

from rlc.envs.continuous_microbe import ContinuousMicrobeEnv, _DELTA_DIR


class WindyMicrobeEnv(ContinuousMicrobeEnv):
    """Continuous microbe environment with a uniform drift field.

    Parameters
    ----------
    current : tuple of float
        The constant velocity field (c_x, c_y) added to the displacement
        at every step, before the Gaussian noise term. Default (0, 0),
        which makes the dynamics identical to those of
        ``ContinuousMicrobeEnv``.
    **kwargs
        Forwarded to ``ContinuousMicrobeEnv``.
    """

    def __init__(
        self,
        current: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.current = np.asarray(current, dtype=np.float64).reshape(2)

    def _intended_landing(self, pos: np.ndarray, action: int) -> np.ndarray:
        """Apply the displacement plus drift plus optional noise, then clip."""
        delta = self.step_displacement * _DELTA_DIR[action] + self.current
        if self.noise > 0:
            delta = delta + self.np_random.normal(0.0, self.noise, size=2)
        landing = pos + delta
        landing[0] = np.clip(landing[0], 0.0, self.width)
        landing[1] = np.clip(landing[1], 0.0, self.height)
        return landing