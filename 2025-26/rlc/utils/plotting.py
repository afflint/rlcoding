"""Plotting utilities for tabular RL experiments.

Functions in this module take either a TrainingHistory or a tabular agent
plus an environment, and return a matplotlib Figure. They are designed to
be called from notebooks and to require no information beyond what is
already exposed by the agent and environment APIs of the course.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from rlc.envs.microbe import MicrobeGridEnv
    from rlc.agents.q_learning import QLearningAgent
    from rlc.utils.training import TrainingHistory


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(
    history: "TrainingHistory",
    *,
    smoothing_window: int = 20,
    figsize: tuple[float, float] = (12, 4),
) -> Figure:
    """Plot training return, greedy evaluation return, and epsilon schedule.

    Parameters
    ----------
    history : TrainingHistory
        Output of ``train``.
    smoothing_window : int
        Window size for the moving-average smoothing of the training return.
        Set to 1 to disable smoothing.
    figsize : tuple of float
        Figure size in inches.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- training return --------------------------------------------------
    ax = axes[0]
    returns = np.asarray(history.episode_returns, dtype=np.float64)
    episodes = np.arange(1, len(returns) + 1)
    ax.plot(episodes, returns, color="#bbbbbb", linewidth=0.8,
            label="per-episode")
    if smoothing_window > 1 and len(returns) >= smoothing_window:
        smoothed = np.convolve(
            returns, np.ones(smoothing_window) / smoothing_window, mode="valid"
        )
        ax.plot(
            np.arange(smoothing_window, len(returns) + 1),
            smoothed,
            color="#1f77b4", linewidth=2.0,
            label=f"moving avg (w={smoothing_window})",
        )
    ax.set_xlabel("episode")
    ax.set_ylabel("training return")
    ax.set_title("Training return (behaviour policy)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # --- evaluation return ------------------------------------------------
    ax = axes[1]
    if history.eval_episodes:
        ep = np.asarray(history.eval_episodes)
        mean = np.asarray(history.eval_mean_returns)
        std = np.asarray(history.eval_std_returns)
        ax.plot(ep, mean, color="#2ca02c", linewidth=2.0, marker="o",
                markersize=4, label="mean")
        ax.fill_between(ep, mean - std, mean + std,
                        color="#2ca02c", alpha=0.2, label="±1 std")
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no evaluation data\n(set eval_every in train)",
                ha="center", va="center", transform=ax.transAxes,
                color="gray", fontsize=10)
    ax.set_xlabel("episode")
    ax.set_ylabel("evaluation return")
    ax.set_title("Greedy evaluation return")
    ax.grid(alpha=0.3)

    # --- epsilon schedule -------------------------------------------------
    ax = axes[2]
    if history.epsilons:
        ax.plot(np.arange(1, len(history.epsilons) + 1), history.epsilons,
                color="#d62728", linewidth=2.0)
    else:
        ax.text(0.5, 0.5, "agent has no epsilon attribute",
                ha="center", va="center", transform=ax.transAxes,
                color="gray", fontsize=10)
    ax.set_xlabel("episode")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_title(r"Exploration schedule ($\varepsilon$)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Policy and value visualization on a grid environment
# ---------------------------------------------------------------------------

# Action -> (dx, dy) for arrow plotting (in matplotlib axis coordinates,
# where dy is positive upward but rows in our grid are positive downward).
_ARROW_DELTA = {
    0: (0.0, +1.0),   # up    (row decreases -> in axis coords dy positive)
    1: (+1.0, 0.0),   # right
    2: (0.0, -1.0),   # down
    3: (-1.0, 0.0),   # left
}


def plot_grid_policy_and_values(
    agent: "QLearningAgent",
    env: "MicrobeGridEnv",
    *,
    figsize: tuple[float, float] = (10, 4),
    cmap: str = "viridis",
    arrow_color: str = "white",
    annotate_values: bool = False,
) -> Figure:
    """Plot V(s) as a heatmap on the grid, with greedy-policy arrows on top.

    Toxic and goal cells are masked out of the value heatmap and visually
    highlighted with their own colors. Start cell is marked with a square.

    Parameters
    ----------
    agent : QLearningAgent
        Trained agent with ``state_values`` and ``greedy_policy`` methods.
    env : MicrobeGridEnv
        The environment whose grid layout is used. The function reads
        ``env.height``, ``env.width``, ``env.start_pos``, ``env.goal_pos``,
        and ``env.toxic_cells``.
    figsize : tuple of float
        Figure size in inches.
    cmap : str
        Matplotlib colormap for the value heatmap.
    arrow_color : str
        Color of the policy arrows.
    annotate_values : bool
        If True, write the numeric value of V(s) inside each free cell.
    """
    H, W = env.height, env.width
    values = agent.state_values().reshape(H, W)
    policy = agent.greedy_policy().reshape(H, W)

    # Mask toxic and goal cells out of the value heatmap.
    mask = np.zeros((H, W), dtype=bool)
    for (r, c) in env.toxic_cells:
        mask[r, c] = True
    gr, gc = env.goal_pos
    mask[gr, gc] = True
    values_masked = np.ma.array(values, mask=mask)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(values_masked, cmap=cmap, origin="upper",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"$V(s) = \max_a Q(s,a)$")

    # Highlight toxic cells (purple) and goal (green); mark start with a square.
    for (r, c) in env.toxic_cells:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   facecolor="#8c2a8c", edgecolor="black",
                                   linewidth=0.5))
    ax.add_patch(plt.Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                               facecolor="#338c33", edgecolor="black",
                               linewidth=0.5))
    sr, sc = env.start_pos
    ax.add_patch(plt.Rectangle((sc - 0.5, sr - 0.5), 1, 1,
                               fill=False, edgecolor="white", linewidth=2.0))

    # Greedy policy arrows on free cells only.
    for r in range(H):
        for c in range(W):
            if mask[r, c] or (r, c) == env.start_pos and (r, c) in env.toxic_cells:
                continue
            if (r, c) == env.goal_pos:
                continue
            if (r, c) in env.toxic_cells:
                continue
            dx, dy = _ARROW_DELTA[int(policy[r, c])]
            ax.arrow(c, r, dx * 0.3, -dy * 0.3,  # invert dy: imshow rows go down
                     head_width=0.18, head_length=0.18,
                     fc=arrow_color, ec=arrow_color, length_includes_head=True)
            if annotate_values:
                ax.text(c, r + 0.35, f"{values[r, c]:.0f}",
                        ha="center", va="center",
                        color=arrow_color, fontsize=7)

    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Greedy policy and state values")
    fig.tight_layout()
    return fig