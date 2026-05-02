"""Microbe grid-world environment.

A small, fully-discrete environment used as the running example in Lecture 1
of the course. A microbe lives on a rectangular grid, starts from a fixed
"safe" cell, and must reach a "nutrient" cell while avoiding "toxic" cells.
Stepping on a toxic cell teleports the microbe back to the start with a
heavy penalty; reaching the nutrient terminates the episode.

The default configuration matches the Cliff Walking benchmark of
Sutton & Barto (Reinforcement Learning: An Introduction, Section 6.5).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Action encoding shared with Gymnasium's CliffWalking convention.
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3

# Row-column displacement for each action.
_DELTA = {
    ACTION_UP: (-1, 0),
    ACTION_RIGHT: (0, +1),
    ACTION_DOWN: (+1, 0),
    ACTION_LEFT: (0, -1),
}

# For each action, its two perpendicular actions (used in slippery dynamics).
_PERPENDICULAR = {
    ACTION_UP: (ACTION_LEFT, ACTION_RIGHT),
    ACTION_RIGHT: (ACTION_UP, ACTION_DOWN),
    ACTION_DOWN: (ACTION_LEFT, ACTION_RIGHT),
    ACTION_LEFT: (ACTION_UP, ACTION_DOWN),
}


class MicrobeGridEnv(gym.Env):
    """A grid-world MDP modelling a microbe in a hostile habitat.

    Parameters
    ----------
    height, width : int
        Grid dimensions. Defaults to 4 x 12 (Cliff Walking layout).
    start_pos : tuple of int
        (row, col) of the start cell.
    goal_pos : tuple of int
        (row, col) of the nutrient cell. Reaching it terminates the episode.
    toxic_cells : iterable of (int, int), optional
        Cells that, when entered, teleport the microbe back to start.
        Defaults to a single row of toxic cells along the bottom edge,
        between start and goal (Cliff Walking layout).
    slippery : float in [0, 1]
        Probability that the commanded action is replaced by one of its
        two perpendicular actions, each with probability slippery / 2.
        Default 0.0 (deterministic).
    step_cost : float
        Reward emitted on every non-toxic transition. Default -1.
    toxic_penalty : float
        Reward emitted when the microbe enters a toxic cell, before being
        teleported back to the start cell. Default -100.
    render_mode : str, optional
        Either ``"rgb_array"`` or ``None`` (no rendering).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        height: int = 4,
        width: int = 12,
        start_pos: tuple[int, int] = (3, 0),
        goal_pos: tuple[int, int] = (3, 11),
        toxic_cells: Optional[list[tuple[int, int]]] = None,
        slippery: float = 0.0,
        step_cost: float = -1.0,
        toxic_penalty: float = -100.0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if toxic_cells is None:
            toxic_cells = [(3, c) for c in range(1, 11)]

        self.height = int(height)
        self.width = int(width)
        self.start_pos = tuple(start_pos)
        self.goal_pos = tuple(goal_pos)
        self.toxic_cells = set(map(tuple, toxic_cells))
        self.slippery = float(slippery)
        self.step_cost = float(step_cost)
        self.toxic_penalty = float(toxic_penalty)

        self._validate_layout()

        self.observation_space = spaces.Discrete(self.height * self.width)
        self.action_space = spaces.Discrete(4)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode={render_mode!r}. "
                f"Supported: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

        # Episode state.
        self._agent_pos: Optional[tuple[int, int]] = None
        self._episode_return: float = 0.0

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _validate_layout(self) -> None:
        def in_bounds(pos: tuple[int, int]) -> bool:
            r, c = pos
            return 0 <= r < self.height and 0 <= c < self.width

        if not in_bounds(self.start_pos):
            raise ValueError(f"start_pos {self.start_pos} is out of bounds.")
        if not in_bounds(self.goal_pos):
            raise ValueError(f"goal_pos {self.goal_pos} is out of bounds.")
        for cell in self.toxic_cells:
            if not in_bounds(cell):
                raise ValueError(f"toxic cell {cell} is out of bounds.")
        if self.start_pos in self.toxic_cells:
            raise ValueError("start_pos cannot be a toxic cell.")
        if self.goal_pos in self.toxic_cells:
            raise ValueError("goal_pos cannot be a toxic cell.")
        if self.start_pos == self.goal_pos:
            raise ValueError("start_pos and goal_pos must differ.")
        if not 0.0 <= self.slippery <= 1.0:
            raise ValueError(f"slippery must be in [0, 1], got {self.slippery}.")

    def _pos_to_obs(self, pos: tuple[int, int]) -> int:
        """Row-major flattening: s = row * width + col."""
        return pos[0] * self.width + pos[1]

    def _intended_landing(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        """Apply action displacement, clipping at grid boundaries."""
        dr, dc = _DELTA[action]
        r = max(0, min(self.height - 1, pos[0] + dr))
        c = max(0, min(self.width - 1, pos[1] + dc))
        return (r, c)

    def _sample_effective_action(self, action: int) -> int:
        """Apply slippery dynamics: with probability `slippery` replace the
        commanded action with one of its two perpendiculars (each p/2)."""
        if self.slippery == 0.0:
            return action
        if self.np_random.random() < self.slippery:
            perp = _PERPENDICULAR[action]
            return int(self.np_random.choice(perp))
        return action

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        self._agent_pos = self.start_pos
        self._episode_return = 0.0
        obs = self._pos_to_obs(self._agent_pos)
        return obs, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self._agent_pos is None:
            raise RuntimeError("step() called before reset().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected 0..3.")

        effective_action = self._sample_effective_action(action)
        landing = self._intended_landing(self._agent_pos, effective_action)

        if landing == self.goal_pos:
            self._agent_pos = landing
            reward = self.step_cost
            terminated = True
        elif landing in self.toxic_cells:
            self._agent_pos = self.start_pos
            reward = self.toxic_penalty
            terminated = False
        else:
            self._agent_pos = landing
            reward = self.step_cost
            terminated = False

        truncated = False
        self._episode_return += reward
        obs = self._pos_to_obs(self._agent_pos)
        info = {"effective_action": effective_action}
        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------

    # Color palette (RGB in [0, 1]).
    _COLOR_FREE = np.array([0.96, 0.96, 0.92])    # pale background
    _COLOR_TOXIC = np.array([0.55, 0.10, 0.55])   # purple band
    _COLOR_GOAL = np.array([0.20, 0.55, 0.20])    # nutrient green
    _COLOR_START = np.array([0.85, 0.85, 0.70])   # faint marker
    _COLOR_GRID = np.array([0.30, 0.30, 0.30])    # grid lines

    # Microbe color endpoints: from "healthy" to "depleted".
    _MICROBE_HEALTHY = np.array([0.10, 0.75, 0.20])  # vivid green
    _MICROBE_DEPLETED = np.array([0.55, 0.05, 0.05])  # dark red
    _ENERGY_DEPLETION_SCALE = 100.0  # cumulative cost at which the microbe
                                     # is considered fully depleted

    _CELL_PIXELS = 32
    _MARGIN = 1

    def render(self) -> Optional[np.ndarray]:
        """Return an RGB image of the current grid as a NumPy array.

        The microbe is drawn as a filled disc whose color interpolates
        between healthy green and depleted red as a function of the
        cumulative reward collected since the last reset. Concretely,
        ``alpha = clip(-episode_return / _ENERGY_DEPLETION_SCALE, 0, 1)``
        and the microbe color is ``(1 - alpha) * healthy + alpha * depleted``.
        """
        if self.render_mode != "rgb_array":
            return None
        if self._agent_pos is None:
            raise RuntimeError("render() called before reset().")

        cell = self._CELL_PIXELS
        m = self._MARGIN
        h_px = self.height * cell
        w_px = self.width * cell

        # Background
        img = np.tile(self._COLOR_FREE, (h_px, w_px, 1))

        # Paint special cells.
        def paint_cell(pos: tuple[int, int], color: np.ndarray) -> None:
            r, c = pos
            r0, r1 = r * cell + m, (r + 1) * cell - m
            c0, c1 = c * cell + m, (c + 1) * cell - m
            img[r0:r1, c0:c1] = color

        paint_cell(self.start_pos, self._COLOR_START)
        for tc in self.toxic_cells:
            paint_cell(tc, self._COLOR_TOXIC)
        paint_cell(self.goal_pos, self._COLOR_GOAL)

        # Grid lines.
        for r in range(self.height + 1):
            img[min(r * cell, h_px - 1), :] = self._COLOR_GRID
        for c in range(self.width + 1):
            img[:, min(c * cell, w_px - 1)] = self._COLOR_GRID

        # Microbe color from cumulative episode return.
        # episode_return is non-positive, so -episode_return >= 0.
        alpha = float(np.clip(-self._episode_return / self._ENERGY_DEPLETION_SCALE,
                              0.0, 1.0))
        microbe_color = (1.0 - alpha) * self._MICROBE_HEALTHY + alpha * self._MICROBE_DEPLETED

        # Draw the microbe as a filled disc inside its current cell.
        ar, ac = self._agent_pos
        cy = ar * cell + cell // 2
        cx = ac * cell + cell // 2
        radius = cell // 2 - 2 * m
        yy, xx = np.ogrid[:h_px, :w_px]
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        img[disc] = microbe_color

        return (img * 255).astype(np.uint8)

    def close(self) -> None:
        # No external resources to release.
        pass