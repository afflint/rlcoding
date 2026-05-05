"""Tabular Monte Carlo control agent.

Unlike the TD methods of the course, the agent does not update its Q-table
on a per-step basis: there is no learning target available until an episode
terminates and the actual returns are observed. The standard agent
interface is therefore reinterpreted:

    update(state, action, reward, next_state, terminated, *, next_action=None)
        Records the (state, action, reward) triple in an episode buffer.
        Does *not* modify the Q-table.

    end_episode()
        Computes the per-step returns by backward recursion, applies the
        Monte Carlo update to each visited (state, action) pair, decays
        epsilon. This is where actual learning happens.

The class supports both first-visit and every-visit Monte Carlo (selected
via the ``every_visit`` flag) and either a 1/N running-mean step size or
a constant step size (selected via ``alpha``: ``None`` for running mean).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class MonteCarloAgent:
    """Tabular on-policy Monte Carlo control agent.

    Parameters
    ----------
    n_states, n_actions : int
        Sizes of the state and action spaces.
    gamma : float
        Discount factor in [0, 1].
    epsilon_start, epsilon_min : float
        Initial and floor values of the epsilon-greedy exploration rate.
    epsilon_decay : float
        Multiplicative decay applied to ``epsilon`` at the end of each
        episode.
    every_visit : bool
        If True, every occurrence of (s, a) within an episode contributes
        to the running average (every-visit MC). If False, only the first
        occurrence contributes (first-visit MC). Default False.
    alpha : float, optional
        If None (default), use the running-mean step size 1 / N(s, a),
        which corresponds to the textbook MC estimator. If a positive
        float is given, use it as a constant step size — useful when the
        policy is non-stationary (typical in MC control with epsilon-greedy
        exploration).
    initial_q : float
        Constant value used to initialise the Q-table.
    seed : int, optional
        Seed of the agent's internal random number generator.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        every_visit: bool = False,
        alpha: Optional[float] = None,
        initial_q: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
        if not 0.0 <= epsilon_min <= epsilon_start <= 1.0:
            raise ValueError(
                f"epsilon range must satisfy 0 <= epsilon_min <= epsilon_start <= 1, "
                f"got start={epsilon_start}, min={epsilon_min}."
            )
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError(f"epsilon_decay must be in (0, 1], got {epsilon_decay}.")
        if alpha is not None and not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha, when not None, must be in (0, 1], got {alpha}.")

        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.every_visit = bool(every_visit)
        self.alpha = None if alpha is None else float(alpha)
        self.initial_q = float(initial_q)

        self.Q = np.full((self.n_states, self.n_actions), self.initial_q, dtype=np.float64)
        # Visit counts per (state, action), used for the 1/N running mean.
        self.N = np.zeros((self.n_states, self.n_actions), dtype=np.int64)

        self.epsilon = self.epsilon_start

        # Per-episode buffer of (state, action, reward) triples.
        self._episode: list[tuple[int, int, float]] = []

        self._rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------------------

    def select_action(self, state: int, *, greedy: bool = False) -> int:
        """Pick an action in ``state`` using epsilon-greedy (or greedy)."""
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return self._argmax_random_tiebreak(self.Q[state])

    def _argmax_random_tiebreak(self, values: np.ndarray) -> int:
        """Argmax with uniform random tie-breaking among maximisers."""
        max_value = values.max()
        candidates = np.flatnonzero(values == max_value)
        return int(self._rng.choice(candidates))

    # ---------------------------------------------------------------------
    # Learning
    # ---------------------------------------------------------------------

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        *,
        next_action: Optional[int] = None,
    ) -> None:
        """Record the (state, action, reward) triple in the episode buffer.

        Monte Carlo learning happens at the end of the episode, not here:
        this method only stores the transition. The ``next_state``,
        ``terminated``, and ``next_action`` arguments are accepted for
        interface uniformity with TD agents and are ignored.
        """
        self._episode.append((int(state), int(action), float(reward)))

    def end_episode(self) -> None:
        """Apply the Monte Carlo update from the buffered trajectory.

        Walks the trajectory backwards to compute G_t, then averages G_t
        into Q[s, a] using either a running mean (alpha=None) or a
        constant step size (alpha=value). With first-visit MC, only the
        first occurrence of each (s, a) within the episode contributes.
        Finally decays epsilon.
        """
        # Walk backwards: G_t = r_{t+1} + gamma * G_{t+1}
        G = 0.0
        # If first-visit, mark which (s, a) pairs we've already seen in the
        # *forward* direction; equivalently, when iterating backwards, the
        # *last* index where (s, a) appears in the backward iteration is
        # the *first* occurrence in forward time. We pre-compute the set
        # of first-visit indices for clarity.
        if self.every_visit or not self._episode:
            indices_to_use = range(len(self._episode))
        else:
            indices_to_use = self._first_visit_indices()

        # We walk all timesteps backwards to compute G; we only apply
        # updates at the indices flagged for use.
        use_set = set(indices_to_use)
        for t in range(len(self._episode) - 1, -1, -1):
            s, a, r = self._episode[t]
            G = r + self.gamma * G
            if t in use_set:
                self.N[s, a] += 1
                step_size = (
                    self.alpha if self.alpha is not None else 1.0 / self.N[s, a]
                )
                self.Q[s, a] += step_size * (G - self.Q[s, a])

        # Reset the episode buffer.
        self._episode.clear()

        # Decay epsilon.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _first_visit_indices(self) -> list[int]:
        """Return the indices of the first occurrence of each (s, a) pair."""
        seen: set[tuple[int, int]] = set()
        first_indices: list[int] = []
        for t, (s, a, _) in enumerate(self._episode):
            key = (s, a)
            if key not in seen:
                seen.add(key)
                first_indices.append(t)
        return first_indices

    # ---------------------------------------------------------------------
    # Convenience views
    # ---------------------------------------------------------------------

    def greedy_policy(self) -> np.ndarray:
        """Return the greedy policy as an array of shape (n_states,)."""
        return np.array(
            [self._argmax_random_tiebreak(self.Q[s]) for s in range(self.n_states)],
            dtype=np.int64,
        )

    def state_values(self) -> np.ndarray:
        """Return V(s) = max_a Q(s, a) as an array of shape (n_states,)."""
        return self.Q.max(axis=1)