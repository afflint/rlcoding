"""Tabular Q-learning agent.

The agent maintains a Q-table of shape (n_states, n_actions) and updates it
on each environment transition with the standard Q-learning rule

    Q[s, a] <- Q[s, a] + alpha * (r + gamma * (1 - terminated) * max Q[s', .] - Q[s, a]).

Action selection is epsilon-greedy with an exponential decay of epsilon over
episodes. The interface (select_action, update, end_episode) is shared with
the SARSA and Monte Carlo agents introduced in Lecture 2, so that the same
training loop drives all three algorithms unchanged.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent for environments with discrete state and
    action spaces.

    Parameters
    ----------
    n_states, n_actions : int
        Sizes of the state and action spaces. For a Gymnasium environment
        with ``Discrete`` spaces these are ``env.observation_space.n`` and
        ``env.action_space.n``.
    alpha : float
        Learning rate (step size) of the Q-learning update.
    gamma : float
        Discount factor in [0, 1].
    epsilon_start, epsilon_min : float
        Initial and floor values of the epsilon-greedy exploration rate.
    epsilon_decay : float
        Multiplicative decay applied to ``epsilon`` at the end of each
        episode: ``epsilon <- max(epsilon_min, epsilon * epsilon_decay)``.
    initial_q : float
        Constant value used to initialise the Q-table. ``0.0`` is the
        textbook choice; positive values implement *optimistic
        initialisation*, which encourages early exploration.
    seed : int, optional
        Seed of the agent's internal random number generator, used for
        epsilon-greedy tie-breaking and exploratory action sampling.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        initial_q: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
        if not 0.0 <= epsilon_min <= epsilon_start <= 1.0:
            raise ValueError(
                f"epsilon range must satisfy 0 <= epsilon_min <= epsilon_start <= 1, "
                f"got start={epsilon_start}, min={epsilon_min}."
            )
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError(f"epsilon_decay must be in (0, 1], got {epsilon_decay}.")

        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.initial_q = float(initial_q)

        # Q-table: rows = states, columns = actions.
        self.Q = np.full((self.n_states, self.n_actions), self.initial_q, dtype=np.float64)

        # Current epsilon (decayed across episodes).
        self.epsilon = self.epsilon_start

        # Internal RNG, separate from the environment's RNG.
        self._rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------------------

    def select_action(self, state: int, *, greedy: bool = False) -> int:
        """Pick an action in ``state``.

        With ``greedy=False`` (default) the action is epsilon-greedy with
        respect to the current Q-table; with ``greedy=True`` the action is
        always the greedy one. Greedy mode is intended for *evaluation*
        rollouts, not for training.
        """
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
    ) -> None:
        """Apply the Q-learning update for a single transition.

        The bootstrap term is zeroed out only when ``terminated`` is True;
        a ``truncated`` transition (not exposed here) should still bootstrap.
        """
        bootstrap = 0.0 if terminated else float(self.Q[next_state].max())
        td_target = reward + self.gamma * bootstrap
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def end_episode(self) -> None:
        """Hook called by the training loop at the end of every episode.

        For Q-learning the only per-episode bookkeeping is the epsilon decay.
        SARSA and Monte Carlo will reuse this hook for their own bookkeeping.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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