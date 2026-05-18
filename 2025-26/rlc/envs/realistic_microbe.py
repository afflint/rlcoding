"""Realistic microbe environment — the showcase environment for Lecture 5 (DQN).

A successor to ``ContinuousMicrobeEnv`` of Lectures 1-4. The microbe still lives
in a 2D Petri dish and still seeks a goal, but the problem is richer in three
ways, each chosen to make a deep Q-network the natural tool:

* **Inertia.** The state is four-dimensional — position *and* velocity,
  ``(x, y, vx, vy)``. The microbe drifts: actions are thrusts, not
  displacements, and a viscous drag damps the velocity each step. The right
  action depends jointly on where the microbe is and how it is moving, so the
  value function genuinely couples all four coordinates.
* **A continuous danger gradient.** Instead of a step-discontinuous toxic
  rectangle with a teleport, the dish carries a smooth toxin concentration
  field — a superposition of Gaussian sources. The microbe is never blocked or
  killed; high-toxin regions simply bleed reward continuously, so the agent
  feels a gradient and learns to weave a low-toxin route between the sources.
* **A genuine positive terminal bonus.** Reaching the goal pays a large
  positive reward, so terminating is unambiguously optimal — unlike the old
  env, whose goal paid only the step cost.

The environment is deterministic and exposes a finite ``observation_space``
Box, so the ``NormalizeObs`` wrapper can rescale observations to [0, 1].
Episodes terminate only on reaching the goal; the env never sets ``truncated``
(the training harness caps episode length).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Action encoding shared with the rest of the course. Actions are now thrust
# directions (accelerations), not displacements.
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3

# Unit thrust vectors in (x, y) coordinates. As in ContinuousMicrobeEnv, y
# increases downward (matching imshow), so "up" decreases y.
_THRUST_DIR = {
    ACTION_UP:    np.array([0.0, -1.0], dtype=np.float64),
    ACTION_RIGHT: np.array([+1.0, 0.0], dtype=np.float64),
    ACTION_DOWN:  np.array([0.0, +1.0], dtype=np.float64),
    ACTION_LEFT:  np.array([-1.0, 0.0], dtype=np.float64),
}


class RealisticMicrobeEnv(gym.Env):
    """Continuous 4D-state, discrete-action microbe environment with inertia.

    Parameters
    ----------
    width, height : float
        Size of the Petri dish.
    start_pos : tuple of float
        (x, y) of the start position. The microbe starts at rest.
    goal_center, goal_radius : tuple, float
        Centre and radius of the disc-shaped goal. Reaching it terminates the
        episode and pays ``goal_bonus``.
    toxic_centers : sequence of (x, y)
        Centres of the Gaussian toxin sources. The concentration field is their
        (clipped) superposition.
    toxic_sigma : float
        Standard deviation (width) of each Gaussian toxin source.
    drag : float
        Fraction of velocity lost to viscous drag each step, in (0, 1).
    thrust : float
        Magnitude of the per-action thrust (acceleration).
    step_cost : float
        Reward emitted every step (the microbe's metabolic cost). Negative.
    toxic_penalty : float
        Weight of the continuous toxin penalty: a step at concentration c emits
        an additional ``-toxic_penalty * c``. Non-negative.
    goal_bonus : float
        Positive reward added on the step that reaches the goal.
    render_mode : str, optional
        Either "rgb_array" or None.

    Notes
    -----
    The maximum per-component speed is ``thrust / drag`` (the drag-limited
    terminal velocity); the observation space uses it as the velocity bound.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(
        self,
        width: float = 12.0,
        height: float = 6.0,
        start_pos: tuple[float, float] = (1.0, 3.0),
        goal_center: tuple[float, float] = (11.0, 3.0),
        goal_radius: float = 0.6,
        toxic_centers: tuple[tuple[float, float], ...] = (
            (4.0, 2.0), (6.5, 4.0), (9.0, 2.0),
        ),
        toxic_sigma: float = 1.1,
        drag: float = 0.2,
        thrust: float = 0.25,
        step_cost: float = -1.0,
        toxic_penalty: float = 8.0,
        goal_bonus: float = 100.0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.width = float(width)
        self.height = float(height)
        self.start_pos = np.array(start_pos, dtype=np.float64)
        self.goal_center = np.array(goal_center, dtype=np.float64)
        self.goal_radius = float(goal_radius)
        self.toxic_centers = tuple((float(cx), float(cy))
                                   for cx, cy in toxic_centers)
        self.toxic_sigma = float(toxic_sigma)
        self.drag = float(drag)
        self.thrust = float(thrust)
        self.step_cost = float(step_cost)
        self.toxic_penalty = float(toxic_penalty)
        self.goal_bonus = float(goal_bonus)

        # Drag-limited terminal velocity: under constant thrust, the recurrence
        # v <- (1 - drag) v + thrust has fixed point thrust / drag, approached
        # monotonically from below, so |v| never exceeds it.
        self.v_max = self.thrust / self.drag

        self._validate_layout()

        # Observation: (x, y, vx, vy) — position in the dish, velocity bounded
        # by the terminal velocity.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -self.v_max, -self.v_max],
                         dtype=np.float32),
            high=np.array([self.width, self.height, self.v_max, self.v_max],
                          dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        if (render_mode is not None
                and render_mode not in self.metadata["render_modes"]):
            raise ValueError(
                f"Unsupported render_mode={render_mode!r}. "
                f"Supported: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

        # Episode state.
        self._agent_pos: Optional[np.ndarray] = None
        self._agent_vel: Optional[np.ndarray] = None
        self._episode_return: float = 0.0

        # Static rendering background (toxin field + goal + start); built once.
        self._background: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _validate_layout(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive.")
        if not (0.0 < self.drag < 1.0):
            raise ValueError(f"drag must lie in (0, 1), got {self.drag}.")
        if self.thrust <= 0:
            raise ValueError(f"thrust must be positive, got {self.thrust}.")
        if self.toxic_sigma <= 0:
            raise ValueError(
                f"toxic_sigma must be positive, got {self.toxic_sigma}.")
        if self.goal_radius <= 0:
            raise ValueError(
                f"goal_radius must be positive, got {self.goal_radius}.")
        if self.toxic_penalty < 0:
            raise ValueError("toxic_penalty must be non-negative.")
        for name, p in (("start_pos", self.start_pos),
                        ("goal_center", self.goal_center)):
            if not (0 <= p[0] <= self.width and 0 <= p[1] <= self.height):
                raise ValueError(f"{name} {tuple(p)} is out of bounds.")
        for cx, cy in self.toxic_centers:
            if not (0 <= cx <= self.width and 0 <= cy <= self.height):
                raise ValueError(
                    f"toxic center ({cx}, {cy}) is out of bounds.")
        # The start must lie outside the goal, and neither start nor goal may
        # be buried in toxin (a misplaced source would make the task ill-posed).
        if self._is_goal(self.start_pos):
            raise ValueError("start_pos must lie outside the goal region.")
        for name, p in (("start_pos", self.start_pos),
                        ("goal_center", self.goal_center)):
            c = self.toxin_concentration(p)
            if c > 0.5:
                raise ValueError(
                    f"{name} sits in high toxin concentration ({c:.2f}); "
                    f"move it or the toxin sources.")

    def _is_goal(self, pos: np.ndarray) -> bool:
        """True if the point ``pos`` lies within the goal disc."""
        return bool(np.linalg.norm(pos - self.goal_center) <= self.goal_radius)

    def _segment_hits_goal(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        """True if the segment p0 -> p1 passes within ``goal_radius`` of the
        goal centre. Using the whole segment (not just the landing point)
        prevents a fast microbe from tunnelling through the goal disc."""
        d = p1 - p0
        seg_len_sq = float(d @ d)
        if seg_len_sq == 0.0:
            closest = p0
        else:
            t = float((self.goal_center - p0) @ d) / seg_len_sq
            t = min(1.0, max(0.0, t))
            closest = p0 + t * d
        return bool(np.linalg.norm(closest - self.goal_center)
                    <= self.goal_radius)

    def toxin_concentration(self, pos) -> float:
        """Toxin concentration in [0, 1] at a single position ``pos = (x, y)``.

        The field is the superposition of the Gaussian sources, clipped to 1.
        """
        x, y = float(pos[0]), float(pos[1])
        two_sigma_sq = 2.0 * self.toxic_sigma ** 2
        c = 0.0
        for cx, cy in self.toxic_centers:
            c += math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / two_sigma_sq)
        return float(min(1.0, c))

    def _obs(self) -> np.ndarray:
        return np.array([self._agent_pos[0], self._agent_pos[1],
                         self._agent_vel[0], self._agent_vel[1]],
                        dtype=np.float32)

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
        self._agent_vel = np.zeros(2, dtype=np.float64)
        self._episode_return = 0.0
        return self._obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._agent_pos is None:
            raise RuntimeError("step() called before reset().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected 0..3.")

        old_pos = self._agent_pos.copy()

        # Inertial update: thrust accelerates, drag damps, position integrates.
        vel = ((1.0 - self.drag) * self._agent_vel
               + self.thrust * _THRUST_DIR[int(action)])
        vel = np.clip(vel, -self.v_max, self.v_max)
        pos = old_pos + vel

        # Walls are inelastic: clip to the dish and kill the normal velocity.
        for axis, limit in ((0, self.width), (1, self.height)):
            if pos[axis] < 0.0:
                pos[axis] = 0.0
                vel[axis] = 0.0
            elif pos[axis] > limit:
                pos[axis] = limit
                vel[axis] = 0.0

        self._agent_pos = pos
        self._agent_vel = vel

        terminated = self._segment_hits_goal(old_pos, pos)
        concentration = self.toxin_concentration(pos)
        reward = self.step_cost - self.toxic_penalty * concentration
        if terminated:
            reward += self.goal_bonus

        self._episode_return += reward
        info = {"toxin": concentration}
        return self._obs(), float(reward), terminated, False, info

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
    _ENERGY_SCALE = 150.0

    _PIXELS_PER_UNIT = 32

    def _build_background(self) -> np.ndarray:
        """Render the static scene once: the smooth toxin field, the goal disc,
        the start marker and the border. The microbe is drawn on top per frame.
        """
        ppu = self._PIXELS_PER_UNIT
        h_px, w_px = int(self.height * ppu), int(self.width * ppu)

        # Pixel-centre coordinates in dish units.
        xs = (np.arange(w_px) + 0.5) / ppu
        ys = (np.arange(h_px) + 0.5) / ppu
        gx, gy = np.meshgrid(xs, ys)

        # Toxin concentration field, vectorised over the pixel grid.
        two_sigma_sq = 2.0 * self.toxic_sigma ** 2
        field = np.zeros((h_px, w_px), dtype=np.float64)
        for cx, cy in self.toxic_centers:
            field += np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / two_sigma_sq)
        field = np.clip(field, 0.0, 1.0)

        # Blend free -> toxic by concentration: the danger is a smooth gradient.
        img = (self._COLOR_FREE[None, None, :]
               + field[:, :, None]
               * (self._COLOR_TOXIC - self._COLOR_FREE)[None, None, :])
        img = np.ascontiguousarray(img)

        # Goal disc.
        goal_mask = ((gx - self.goal_center[0]) ** 2
                     + (gy - self.goal_center[1]) ** 2) <= self.goal_radius ** 2
        img[goal_mask] = self._COLOR_GOAL

        # Start marker (small square outline).
        sx, sy = int(self.start_pos[0] * ppu), int(self.start_pos[1] * ppu)
        s = int(0.4 * ppu)
        sx0, sx1 = sx - s, sx + s
        sy0, sy1 = sy - s, sy + s
        if 0 <= sy0 and sy1 < h_px and 0 <= sx0 and sx1 < w_px:
            img[sy0:sy1, sx0] = self._COLOR_START
            img[sy0:sy1, sx1] = self._COLOR_START
            img[sy0, sx0:sx1] = self._COLOR_START
            img[sy1, sx0:sx1] = self._COLOR_START

        # Outer border.
        img[0, :] = self._COLOR_BORDER
        img[-1, :] = self._COLOR_BORDER
        img[:, 0] = self._COLOR_BORDER
        img[:, -1] = self._COLOR_BORDER

        return img

    def render(self) -> Optional[np.ndarray]:
        """Return an RGB image of the current state of the dish.

        The microbe is a filled disc whose colour interpolates from healthy
        green to depleted red as cumulative episode cost grows.
        """
        if self.render_mode != "rgb_array":
            return None
        if self._agent_pos is None:
            raise RuntimeError("render() called before reset().")

        if self._background is None:
            self._background = self._build_background()
        img = self._background.copy()
        h_px, w_px = img.shape[:2]
        ppu = self._PIXELS_PER_UNIT

        # Microbe colour from cumulative return (the "energy" narrative).
        alpha = float(np.clip(-self._episode_return / self._ENERGY_SCALE,
                              0.0, 1.0))
        microbe_color = ((1.0 - alpha) * self._MICROBE_HEALTHY
                         + alpha * self._MICROBE_DEPLETED)

        ax_px, ay_px = self._agent_pos[0] * ppu, self._agent_pos[1] * ppu
        radius = max(2, int(0.18 * ppu))
        yy, xx = np.ogrid[:h_px, :w_px]
        disc = (yy - ay_px) ** 2 + (xx - ax_px) ** 2 <= radius ** 2
        img[disc] = microbe_color

        return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

    def close(self) -> None:
        pass

class NormalizeObs(gym.ObservationWrapper):
    """Rescale Box observations to [0, 1] using the environment's fixed bounds.

    The Q-network reads the state directly, with no hand-crafted feature map to
    put the coordinates on a common scale, so the first layer and gradient
    descent are sensitive to raw input scale -- position spans 0..12, velocity
    -1.25..1.25. Rescaling every coordinate to [0, 1] removes that sensitivity.
    """

    def __init__(self, env):
        super().__init__(env)
        low, high = env.observation_space.low, env.observation_space.high
        self._low = low
        self._span = high - low
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=low.shape, dtype=np.float32)

    def observation(self, obs):
        return ((obs - self._low) / self._span).astype(np.float32)