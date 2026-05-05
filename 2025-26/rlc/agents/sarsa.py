"""Tabular SARSA agent.

The agent maintains a Q-table of shape (n_states, n_actions) and updates it
on each environment transition with the SARSA rule

    Q[s, a] <- Q[s, a] + alpha * (r + gamma * (1 - terminated) * Q[s', a'] - Q[s, a])

where ``a'`` is the action sampled by the behaviour policy at the new state.
The training loop is responsible for sampling ``a'`` *before* the update and
passing it through the ``next_action`` argument: this preserves the on-policy
property and ensures that the action evaluated in the target is the same
action that will actually be executed at the next step.

The interface (select_action, update, end_episode) is shared with the
Q-learning and Monte Carlo agents of the course. The differences between
SARSA and Q-learning are entirely confined to the body of ``update``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SARSAAgent:
    """Tabular SARSA agent for environments with discrete state and action
    spaces.

    Parameters
    ----------
    n_states, n_actions : int
        Sizes of the state and action spaces.
    alpha : float
        Learning rate of the SARSA update.
    gamma : float
        Discount factor in [0, 1].
    epsilon_start, epsilon_min : float
        Initial and floor values of the epsilon-greedy exploration rate.
    epsilon_decay : float
        Multiplicative decay applied to ``epsilon`` at the end of each
        episode: ``epsilon <- max(epsilon_min, epsilon * epsilon_decay)``.
    initial_q : float
        Constant value used to initialise the Q-table.
    seed : int, optional
        Seed of the agent's internal random number generator.
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

        self.Q = np.full((self.n_states, self.n_actions), self.initial_q, dtype=np.float64)
        self.epsilon = self.epsilon_start

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
        """Apply the SARSA update for a single transition.

        Parameters
        ----------
        state, action : int
            State and action at time t.
        reward : float
            Immediate reward observed.
        next_state : int
            State at time t+1.
        terminated : bool
            Whether ``next_state`` is a terminal state of the MDP.
        next_action : int, optional
            Action sampled by the behaviour policy at ``next_state``.
            Required when ``terminated`` is False; ignored (and may be
            ``None``) when ``terminated`` is True, because the bootstrap
            term is then zero.
        """
        if terminated:
            bootstrap = 0.0
        else:
            if next_action is None:
                raise ValueError(
                    "SARSAAgent.update requires `next_action` for non-terminal "
                    "transitions; got next_action=None with terminated=False."
                )
            bootstrap = float(self.Q[next_state, next_action])

        td_target = reward + self.gamma * bootstrap
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def end_episode(self) -> None:
        """End-of-episode hook: decay epsilon."""
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