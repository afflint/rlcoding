"""Linear semi-gradient SARSA for continuous-state environments.

The agent maintains one weight vector per discrete action and computes the
action-value as the dot product between that weight vector and a feature
representation of the state, produced by an externally-supplied feature
extractor.

Updates follow the on-policy semi-gradient SARSA rule

    w_a <- w_a + alpha [r + gamma q(s', a') - q(s, a)] x(s)

where ``a'`` is the action sampled by the behaviour policy at the new state.
The training loop is responsible for sampling ``a'`` *before* the update and
passing it through the ``next_action`` argument; the same convention is
used by the tabular SARSA agent of Lecture 2 and by the training utilities
of the course.

The interface (select_action, update, end_episode) matches the ``Agent``
protocol shared with the Q-learning and Monte Carlo agents of the course.
The only difference between this agent and ``LinearQLearningAgent`` is
the body of ``update``: SARSA replaces the ``max_a' q(s', a')`` bootstrap
of Q-learning with the value of the action actually sampled at ``s'``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from rlc.utils.features import FeatureExtractor


class LinearSARSAAgent:
    """Semi-gradient SARSA with linear function approximation.

    Parameters
    ----------
    feature_extractor : FeatureExtractor
        Object exposing ``n_features`` and a ``__call__`` that maps a state
        to a feature vector. Determines the representation the agent uses.
    n_actions : int
        Number of discrete actions.
    alpha : float
        Step size of the semi-gradient update. With tile coding, a common
        rescaling is ``alpha / n_tilings`` to keep the per-update magnitude
        comparable to the tabular case; we leave this to the caller.
    gamma : float
        Discount factor in [0, 1].
    epsilon_start, epsilon_min : float
        Initial and floor values of the epsilon-greedy exploration rate.
    epsilon_decay : float
        Multiplicative decay applied to ``epsilon`` at the end of each episode.
    initial_w : float
        Constant value used to initialise every entry of the weight matrix.
        Default 0.0; with binary sparse features this is equivalent to the
        tabular ``initial_q`` of the previous lectures.
    seed : int, optional
        Seed of the agent's internal random number generator, used for
        epsilon-greedy exploration and tie-breaking.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.997,
        initial_w: float = 0.0,
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
        if n_actions < 1:
            raise ValueError(f"n_actions must be positive, got {n_actions}.")

        self.feature_extractor = feature_extractor
        self.n_features = int(feature_extractor.n_features)
        self.n_actions = int(n_actions)

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.initial_w = float(initial_w)

        self.W = np.full((self.n_actions, self.n_features), self.initial_w,
                         dtype=np.float64)
        self.epsilon = self.epsilon_start

        self._rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------------------

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return the vector of Q-values for each action at ``state``."""
        x = self.feature_extractor(state)
        return self.W @ x

    def select_action(self, state: np.ndarray, *, greedy: bool = False) -> int:
        """Pick an action using epsilon-greedy (or greedy if requested)."""
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        q = self.q_values(state)
        return self._argmax_random_tiebreak(q)

    def _argmax_random_tiebreak(self, values: np.ndarray) -> int:
        max_value = values.max()
        candidates = np.flatnonzero(values == max_value)
        return int(self._rng.choice(candidates))

    # ---------------------------------------------------------------------
    # Learning
    # ---------------------------------------------------------------------

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        *,
        next_action: Optional[int] = None,
    ) -> None:
        """Apply the semi-gradient SARSA update for a single transition.

        Parameters
        ----------
        state, action : array-like, int
            State and action at time t.
        reward : float
            Immediate reward observed.
        next_state : array-like
            State at time t+1.
        terminated : bool
            Whether ``next_state`` is a terminal state of the MDP.
        next_action : int, optional
            Action sampled by the behaviour policy at ``next_state``.
            Required when ``terminated`` is False; ignored (and may be
            ``None``) when ``terminated`` is True, because the bootstrap
            term is then zero.
        """
        x = self.feature_extractor(state)
        q_sa = float(self.W[action] @ x)

        if terminated:
            target = float(reward)
        else:
            if next_action is None:
                raise ValueError(
                    "LinearSARSAAgent.update requires `next_action` for "
                    "non-terminal transitions; got next_action=None with "
                    "terminated=False."
                )
            x_next = self.feature_extractor(next_state)
            q_next = float(self.W[next_action] @ x_next)
            target = float(reward) + self.gamma * q_next

        td_error = target - q_sa
        self.W[action] += self.alpha * td_error * x

    def end_episode(self) -> None:
        """Decay epsilon at the end of an episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---------------------------------------------------------------------
    # Convenience views
    # ---------------------------------------------------------------------

    def state_value(self, state: np.ndarray) -> float:
        """Return V(s) = max_a Q(s, a) at a single state."""
        return float(self.q_values(state).max())

    def greedy_action(self, state: np.ndarray) -> int:
        """Return the greedy action at a single state."""
        return self._argmax_random_tiebreak(self.q_values(state))
    

