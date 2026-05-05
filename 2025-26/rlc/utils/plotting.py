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
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
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
    env = getattr(env, "unwrapped", env)
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
                   interpolation="nearest", vmin=vmin, vmax=vmax)
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

# ---------------------------------------------------------------------------
# Visit count visualisations
# ---------------------------------------------------------------------------

def plot_state_visits(
    state_action_counts: np.ndarray,
    env: "MicrobeGridEnv",
    *,
    figsize: tuple[float, float] = (10, 4),
    cmap: str = "magma",
    log_scale: bool = True,
    annotate: bool = False,
) -> Figure:
    """Plot the marginal state-visit count N(s) = sum_a N(s, a) on the grid.

    Parameters
    ----------
    state_action_counts : np.ndarray of int, shape (n_states, n_actions)
        Visit-count matrix as produced by ``VisitCountWrapper``.
    env : MicrobeGridEnv
        Environment whose grid layout is used for display.
    figsize : tuple of float
    cmap : str
    log_scale : bool
        If True, use a log color scale: useful because state-visit counts
        in RL are highly skewed (the path the agent prefers gets visited
        orders of magnitude more often than the corners).
    annotate : bool
        If True, write the count inside each cell.
    """
    env = getattr(env, "unwrapped", env)
    H, W = env.height, env.width
    counts = state_action_counts.sum(axis=1).reshape(H, W).astype(np.float64)

    fig, ax = plt.subplots(figsize=figsize)

    if log_scale:
        # Add 1 inside the log so that zero-visited cells don't blow up.
        display = np.log1p(counts)
        cbar_label = r"$\log_{10}(1 + N(s))$"
        # We technically used log1p (natural log); convert to log10 for a
        # more intuitive colour bar label.
        display = display / np.log(10.0)
    else:
        display = counts
        cbar_label = "$N(s)$"

    im = ax.imshow(display, cmap=cmap, origin="upper", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label)

    # Highlight start, goal, toxic cells with outlines.
    sr, sc = env.start_pos
    gr, gc = env.goal_pos
    ax.add_patch(plt.Rectangle((sc - 0.5, sr - 0.5), 1, 1,
                               fill=False, edgecolor="white", linewidth=2.0))
    ax.add_patch(plt.Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                               fill=False, edgecolor="#338c33", linewidth=2.0))
    for (r, c) in env.toxic_cells:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   fill=False, edgecolor="#8c2a8c", linewidth=1.5))

    if annotate:
        for r in range(H):
            for c in range(W):
                ax.text(c, r, f"{int(counts[r, c])}",
                        ha="center", va="center",
                        color="white", fontsize=7)

    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3)
    ax.tick_params(which="minor", length=0)
    ax.set_title("State visit counts $N(s)$")
    fig.tight_layout()
    return fig


_ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}


def plot_state_action_visits(
    state_action_counts: np.ndarray,
    env: "MicrobeGridEnv",
    *,
    figsize: tuple[float, float] = (12, 6),
    cmap: str = "magma",
    log_scale: bool = True,
    annotate: bool = False,
) -> Figure:
    """Plot per-action visit counts N(s, a) as colored arrows on a single grid.

    For each cell, four arrows are drawn radiating from the cell center
    towards the four cardinal directions. Each arrow is coloured according
    to the count of the corresponding (s, a) pair, on a colour scale shared
    across all arrows in the figure (so that magnitudes are directly
    comparable between cells).

    Cells with N(s, a) = 0 are shown with very faint arrows: this preserves
    the visual fact that an action *was not tried* at that state, instead
    of leaving an ambiguous gap.

    Parameters
    ----------
    state_action_counts : np.ndarray of int, shape (n_states, n_actions)
        Visit-count matrix, with action ordering 0=up, 1=right, 2=down, 3=left.
    env : MicrobeGridEnv
        Environment whose grid layout is used. Wrapped environments are
        accepted: the function strips wrappers via ``unwrapped``.
    figsize : tuple of float
    cmap : str
        Matplotlib colormap.
    log_scale : bool
        If True (recommended), colour intensity scales with log(1 + N(s,a)).
    annotate : bool
        If True, write the integer count next to each arrow. Useful for
        small grids; cluttered for large ones.
    """
    env = getattr(env, "unwrapped", env)
    H, W = env.height, env.width
    n_actions = state_action_counts.shape[1]
    if n_actions != 4:
        raise ValueError(
            f"plot_state_action_visits expects 4 actions, got {n_actions}."
        )

    # Color scaling: log1p compresses the heavy skew, then linear normalise.
    if log_scale:
        values = np.log1p(state_action_counts.astype(np.float64))
        cbar_label = r"$\log_{10}(1 + N(s,a))$"
        # convert from natural log to log10 for the colorbar label semantics
        values = values / np.log(10.0)
    else:
        values = state_action_counts.astype(np.float64)
        cbar_label = "$N(s, a)$"

    vmin, vmax = float(values.min()), float(values.max())
    if vmax == vmin:
        vmax = vmin + 1.0  # avoid degenerate normalisation
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)

    # Light neutral background filling the whole grid.
    ax.imshow(np.zeros((H, W)),
              cmap=plt.cm.gray_r, vmin=0, vmax=1,
              alpha=0.05, origin="upper", interpolation="nearest")

    # Highlight start, goal, and toxic cells with outlines (no fill,
    # to keep the focus on the arrows).
    sr, sc = env.start_pos
    gr, gc = env.goal_pos
    ax.add_patch(plt.Rectangle((sc - 0.5, sr - 0.5), 1, 1,
                               fill=False, edgecolor="black", linewidth=2.0))
    ax.add_patch(plt.Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                               fill=False, edgecolor="#338c33", linewidth=2.0))
    for (r, c) in env.toxic_cells:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   facecolor="#8c2a8c", alpha=0.15,
                                   edgecolor="#8c2a8c", linewidth=1.0))

    # Arrow geometry inside one cell.
    arrow_length = 0.32
    head_width = 0.10
    head_length = 0.08
    # (dx, dy) in axis coordinates: dy positive *upward* on screen, but
    # imshow rows increase downward, so we invert dy when drawing.
    arrow_offsets = {
        0: (0.0, +1.0),   # up:    on grid, row decreases  -> screen dy positive
        1: (+1.0, 0.0),   # right
        2: (0.0, -1.0),   # down:  row increases           -> screen dy negative
        3: (-1.0, 0.0),   # left
    }

    for r in range(H):
        for c in range(W):
            # Skip arrows for terminal cells: from goal no action is ever
            # taken (episode terminates on entry); toxic cells are only ever
            # entered, never acted from (the agent is teleported on entry).
            if (r, c) == env.goal_pos or (r, c) in env.toxic_cells:
                continue

            for a in range(4):
                count_val = state_action_counts[r * W + c, a]
                color = colormap(norm(values[r * W + c, a]))
                # Faint arrows when the (s, a) was never visited.
                alpha = 0.25 if count_val == 0 else 1.0

                dx, dy = arrow_offsets[a]
                # Convert "screen-up" dy to imshow-coordinate dy: invert.
                ax.arrow(
                    c, r,
                    dx * arrow_length, -dy * arrow_length,
                    head_width=head_width, head_length=head_length,
                    fc=color, ec=color, alpha=alpha,
                    length_includes_head=True, linewidth=1.5,
                )
                if annotate and count_val > 0:
                    text_x = c + dx * (arrow_length + 0.12)
                    text_y = r - dy * (arrow_length + 0.12)
                    ax.text(text_x, text_y, f"{int(count_val)}",
                            ha="center", va="center", fontsize=6,
                            color="black")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # invert y so row 0 is on top
    ax.set_aspect("equal")
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3)
    ax.tick_params(which="minor", length=0)
    ax.set_title("State-action visit counts $N(s,a)$  (per-cell directional arrows)")
    fig.tight_layout()
    return fig


def plot_evaluation_curves_overlay(
    histories: dict[str, "TrainingHistory"],
    *,
    figsize: tuple[float, float] = (10, 5),
    colors: Optional[dict[str, str]] = None,
    title: str = "Greedy evaluation return — comparison",
) -> Figure:
    """Overlay greedy evaluation curves of several training runs.

    Each entry in ``histories`` is plotted as a line with a one-standard-
    deviation band, on a shared axis. Useful for side-by-side comparison
    of two or more agents trained on equivalent setups.

    Parameters
    ----------
    histories : dict mapping str -> TrainingHistory
        Keys are display labels (e.g. "Q-learning", "SARSA"); values are
        the corresponding training histories produced by ``train``.
    figsize : tuple of float
    colors : dict, optional
        Mapping from labels to matplotlib colors. If omitted, uses the
        default property cycle.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, (label, hist) in enumerate(histories.items()):
        if not hist.eval_episodes:
            continue
        ep = np.asarray(hist.eval_episodes)
        mean = np.asarray(hist.eval_mean_returns)
        std = np.asarray(hist.eval_std_returns)
        c = (colors or {}).get(label, color_cycle[i % len(color_cycle)])
        ax.plot(ep, mean, color=c, linewidth=2.0, marker="o",
                markersize=4, label=label)
        ax.fill_between(ep, mean - std, mean + std, color=c, alpha=0.15)

    ax.set_xlabel("episode")
    ax.set_ylabel("evaluation return")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def summarize_evaluations(
    histories: dict[str, "TrainingHistory"],
    *,
    last_k: int = 10,
) -> list[dict[str, float]]:
    """Return a list of summary records, one per training run.

    For each run, computes:
    - the final greedy evaluation return (last checkpoint),
    - the standard deviation reported at the final checkpoint
      (across the eval_episodes rollouts of that single checkpoint),
    - the mean and standard deviation of the greedy evaluation return
      over the last ``last_k`` checkpoints, which characterise the
      *long-run* level and stability of the run.

    Parameters
    ----------
    histories : dict mapping str -> TrainingHistory
    last_k : int
        Number of trailing evaluation checkpoints over which to compute
        the long-run mean and standard deviation. If a run has fewer than
        ``last_k`` checkpoints, all available checkpoints are used.

    Returns
    -------
    list of dicts with keys:
        agent, final_mean, final_std, last_k_mean, last_k_std
    """
    out = []
    for label, hist in histories.items():
        if not hist.eval_episodes:
            out.append(dict(
                agent=label,
                final_mean=float("nan"),
                final_std=float("nan"),
                last_k_mean=float("nan"),
                last_k_std=float("nan"),
            ))
            continue

        means = np.asarray(hist.eval_mean_returns, dtype=np.float64)
        stds = np.asarray(hist.eval_std_returns, dtype=np.float64)

        k = min(last_k, len(means))
        last_means = means[-k:]

        out.append(dict(
            agent=label,
            final_mean=float(means[-1]),
            final_std=float(stds[-1]),
            last_k_mean=float(last_means.mean()),
            last_k_std=float(last_means.std()),
        ))
    return out