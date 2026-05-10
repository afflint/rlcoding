"""Continuous-state microbe environment.

A continuous-state version of the microbe grid world used in Lectures 1
and 2. The microbe lives in a 2D rectangle (a Petri dish) and moves with
a fixed displacement per action, optionally perturbed by additive
Gaussian noise on the outcome of the displacement.

The default layout is a 12 x 6 dish with a horizontal toxic rectangle
along the bottom edge and a circular goal at the bottom-right corner —
chosen so that, when the dish is discretized into unit cells, the
resulting tabular environment is essentially identical to
``MicrobeGridEnv`` of Lecture 1.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Action encoding shared with the rest of the course.
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3

# Displacement vectors in (x, y) coordinates.
# Note: y increases downward in our visual convention (matching imshow),
# so "up" decreases y.
_DELTA_DIR = {
    ACTION_UP:    np.array([0.0, -1.0], dtype=np.float64),
    ACTION_RIGHT: np.array([+1.0, 0.0], dtype=np.float64),
    ACTION_DOWN:  np.array([0.0, +1.0], dtype=np.float64),
    ACTION_LEFT:  np.array([-1.0, 0.0], dtype=np.float64),
}


class ContinuousMicrobeEnv(gym.Env):
    """A continuous-state, discrete-action grid-like environment.

    Parameters
    ----------
    width, height : float
        Size of the Petri dish.
    start_pos : tuple of float
        (x, y) of the start position. Must lie outside both regions.
    goal_center, goal_radius : tuple, float
        Centre and radius of the disc-shaped goal region. Reaching it
        terminates the episode.
    toxic_rect : tuple of (xmin, ymin, xmax, ymax)
        Rectangle defining the toxic region. Entering it triggers a
        teleport back to the start, with a heavy reward penalty.
    step_displacement : float
        Magnitude of the per-action displacement, before noise. Default 1.0.
    noise : float
        Standard deviation of the isotropic Gaussian noise added to the
        displacement vector. Default 0.0 (deterministic).
    step_cost : float
        Reward emitted on every non-toxic transition. Default -1.
    toxic_penalty : float
        Reward emitted when the microbe lands in the toxic region.
        Default -100.
    smooth_reward : bool
        If True, replace the discrete reward scheme above with a smooth
        spatial reward field. Reserved for future lectures: setting this
        to True will currently raise NotImplementedError. Default False.
    render_mode : str, optional
        Either ``"rgb_array"`` or ``None`` (no rendering).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(
        self,
        width: float = 12.0,
        height: float = 6.0,
        start_pos: tuple[float, float] = (0.5, 5.5),
        goal_center: tuple[float, float] = (11.5, 5.5),
        goal_radius: float = 0.5,
        toxic_rect: tuple[float, float, float, float] = (1.0, 4.5, 11.0, 5.5),
        step_displacement: float = 1.0,
        noise: float = 0.0,
        step_cost: float = -1.0,
        toxic_penalty: float = -100.0,
        smooth_reward: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if smooth_reward:
            raise NotImplementedError(
                "smooth_reward is reserved for future lectures and is not "
                "implemented yet. Pass smooth_reward=False (the default)."
            )

        self.width = float(width)
        self.height = float(height)
        self.start_pos = np.array(start_pos, dtype=np.float64)
        self.goal_center = np.array(goal_center, dtype=np.float64)
        self.goal_radius = float(goal_radius)
        self.toxic_rect = tuple(map(float, toxic_rect))
        self.step_displacement = float(step_displacement)
        self.noise = float(noise)
        self.step_cost = float(step_cost)
        self.toxic_penalty = float(toxic_penalty)
        self.smooth_reward = bool(smooth_reward)

        self._validate_layout()

        # Observation space: a point in the rectangle [0, W] x [0, H].
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.width, self.height], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode={render_mode!r}. "
                f"Supported: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

        # Episode state
        self._agent_pos: Optional[np.ndarray] = None
        self._episode_return: float = 0.0

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _validate_layout(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width and height must be positive.")
        if not (0 <= self.start_pos[0] <= self.width and
                0 <= self.start_pos[1] <= self.height):
            raise ValueError(f"start_pos {tuple(self.start_pos)} is out of bounds.")
        if not (0 <= self.goal_center[0] <= self.width and
                0 <= self.goal_center[1] <= self.height):
            raise ValueError(f"goal_center {tuple(self.goal_center)} is out of bounds.")
        if self.goal_radius <= 0:
            raise ValueError(f"goal_radius must be positive, got {self.goal_radius}.")
        xmin, ymin, xmax, ymax = self.toxic_rect
        if not (xmin < xmax and ymin < ymax):
            raise ValueError(f"toxic_rect must satisfy xmin<xmax and ymin<ymax.")
        if self._is_toxic(self.start_pos) or self._is_goal(self.start_pos):
            raise ValueError(f"start_pos must lie outside both regions.")
        if self.noise < 0:
            raise ValueError(f"noise must be non-negative.")
        if self.step_displacement <= 0:
            raise ValueError(f"step_displacement must be positive.")

    def _is_goal(self, pos: np.ndarray) -> bool:
        return bool(np.linalg.norm(pos - self.goal_center) <= self.goal_radius)

    def _is_toxic(self, pos: np.ndarray) -> bool:
        xmin, ymin, xmax, ymax = self.toxic_rect
        return bool(xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax)

    def _intended_landing(self, pos: np.ndarray, action: int) -> np.ndarray:
        """Apply the displacement plus optional noise, then clip to the dish."""
        delta = self.step_displacement * _DELTA_DIR[action]
        if self.noise > 0:
            delta = delta + self.np_random.normal(0.0, self.noise, size=2)
        landing = pos + delta
        landing[0] = np.clip(landing[0], 0.0, self.width)
        landing[1] = np.clip(landing[1], 0.0, self.height)
        return landing

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._agent_pos = self.start_pos.copy()
        self._episode_return = 0.0
        return self._agent_pos.astype(np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._agent_pos is None:
            raise RuntimeError("step() called before reset().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected 0..3.")

        landing = self._intended_landing(self._agent_pos, int(action))

        if self._is_goal(landing):
            self._agent_pos = landing
            reward = self.step_cost
            terminated = True
        elif self._is_toxic(landing):
            self._agent_pos = self.start_pos.copy()
            reward = self.toxic_penalty
            terminated = False
        else:
            self._agent_pos = landing
            reward = self.step_cost
            terminated = False

        truncated = False
        self._episode_return += reward
        info = {"landing_before_teleport": landing.copy()}
        return self._agent_pos.astype(np.float32), reward, terminated, truncated, info

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------

    _COLOR_FREE = np.array([0.96, 0.96, 0.92])
    _COLOR_TOXIC = np.array([0.55, 0.10, 0.55])
    _COLOR_GOAL = np.array([0.20, 0.55, 0.20])
    _COLOR_START = np.array([0.85, 0.85, 0.70])
    _COLOR_BORDER = np.array([0.30, 0.30, 0.30])

    _MICROBE_HEALTHY = np.array([0.10, 0.75, 0.20])
    _MICROBE_DEPLETED = np.array([0.55, 0.05, 0.05])
    _ENERGY_DEPLETION_SCALE = 100.0

    _PIXELS_PER_UNIT = 32

    def render(self) -> Optional[np.ndarray]:
        """Return an RGB image of the current state of the dish.

        The microbe is drawn as a filled disc whose color interpolates
        from healthy green to depleted red as a function of cumulative
        episode return.
        """
        if self.render_mode != "rgb_array":
            return None
        if self._agent_pos is None:
            raise RuntimeError("render() called before reset().")

        ppu = self._PIXELS_PER_UNIT
        h_px = int(self.height * ppu)
        w_px = int(self.width * ppu)

        img = np.tile(self._COLOR_FREE, (h_px, w_px, 1))

        # Toxic rectangle.
        xmin, ymin, xmax, ymax = self.toxic_rect
        x0, x1 = int(xmin * ppu), int(xmax * ppu)
        y0, y1 = int(ymin * ppu), int(ymax * ppu)
        img[y0:y1, x0:x1] = self._COLOR_TOXIC

        # Goal disc.
        gy, gx = self.goal_center[1] * ppu, self.goal_center[0] * ppu
        gr = self.goal_radius * ppu
        yy, xx = np.ogrid[:h_px, :w_px]
        goal_mask = (yy - gy) ** 2 + (xx - gx) ** 2 <= gr ** 2
        img[goal_mask] = self._COLOR_GOAL

        # Start marker (small square outline).
        sx, sy = self.start_pos[0] * ppu, self.start_pos[1] * ppu
        s_size = int(0.4 * ppu)
        for d in range(2):
            sy0 = int(sy) - s_size + d
            sy1 = int(sy) + s_size - d
            sx0 = int(sx) - s_size + d
            sx1 = int(sx) + s_size - d
            if 0 <= sy0 < h_px and 0 <= sy1 < h_px and 0 <= sx0 < w_px and 0 <= sx1 < w_px:
                img[sy0, sx0:sx1] = self._COLOR_START
                img[sy1, sx0:sx1] = self._COLOR_START
                img[sy0:sy1, sx0] = self._COLOR_START
                img[sy0:sy1, sx1] = self._COLOR_START

        # Outer border.
        img[0, :] = self._COLOR_BORDER
        img[-1, :] = self._COLOR_BORDER
        img[:, 0] = self._COLOR_BORDER
        img[:, -1] = self._COLOR_BORDER

        # Microbe color from cumulative return.
        alpha = float(np.clip(-self._episode_return / self._ENERGY_DEPLETION_SCALE,
                              0.0, 1.0))
        microbe_color = (1.0 - alpha) * self._MICROBE_HEALTHY + alpha * self._MICROBE_DEPLETED

        # Draw microbe.
        ax_px, ay_px = self._agent_pos[0] * ppu, self._agent_pos[1] * ppu
        radius = max(2, int(0.15 * ppu))
        disc = (yy - ay_px) ** 2 + (xx - ax_px) ** 2 <= radius ** 2
        img[disc] = microbe_color

        return (img * 255).astype(np.uint8)

    def close(self) -> None:
        pass