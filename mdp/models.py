from typing import Tuple, Callable, List, Set
from abc import ABC, abstractmethod

import numpy as np


class MDP(ABC):
    """
    Implements a Markov Decision Process that is organized as follows
    - states: set of available states in the model
    - actions(s): set of available actions from state s
    - T(s, a, s'): transition probability of state s' by taking action a from state s
    - reward(s, a, s'): reward of being is s' after (s, a)
    - is_end(s): states is s is the terminal state
    - gamma: 0 <= gamma <= 1, discount factor
    """

    def __init__(self, gamma: float = 1.):
        self.gamma = gamma

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def states(self) -> Set:
        """
        Returns available states
        :return: set of states
        """

    @abstractmethod
    def actions(self, state) -> Set:
        """
        Returns available actions for state s
        :param state: the state
        :return: set of actions
        """

    @abstractmethod
    def transition(self, start_state, action, end_state) -> float:
        """
        Returns the probability of end_state from (start_state, action)
        :param start_state: the state where we took action from
        :param action: action chosen
        :param end_state: target state
        :return: probability 0 <= p <= 1
        """

    @abstractmethod
    def reward(self, start_state, action, end_state) -> float:
        """
        Numeric reward for (s, a, s')
        :param start_state: start state
        :param action: action
        :param end_state: final state of transition
        :return: reward value
        """

    @abstractmethod
    def end(self, state) -> bool:
        """
        Ending state?
        :param state: state
        :return: true if state is the ending state
        """

    def successors(self, state, action):
        """
        For all the possible states yields a list of successor state with
        transition probability and reward
        :return: [(s_prime, p, r), ...]
        """
        for s_prime in self.states():
            yield s_prime, self.transition(state, action, s_prime), self.reward(state, action, s_prime)

    def transition_reward(self, start_state, action, end_state):
        return self.transition(start_state, action, end_state), \
               self.reward(start_state, action, end_state)


class Policy(ABC):

    def __init__(self, mdp: MDP):
        self.mdp = mdp

    @abstractmethod
    def action(self, state):
        pass

    def episode(self, max_len: int = 1000) -> list:
        """
        Generates a random path over MDP. Raise exception is actions or
        states are not compatible with MDP.
        This is for finite horizon MDPs mainly (see max_len).
        :return: list of (start, action, reward))
        """
        e = []
        current_state = np.random.choice(list(self.mdp.start()))
        while not self.mdp.end(current_state):
            action = self.action(current_state)
            states, probs = [], []
            for candidate_state in self.mdp.states():
                states.append(candidate_state)
                probs.append(self.mdp.transition(current_state, action, candidate_state))
            chosen = np.random.choice(states, p=probs)
            e.append((current_state, action, chosen, self.mdp.reward(
                current_state, action, chosen)))
            current_state = chosen
            if len(e) >= max_len:
                break
        return e

    def utility(self, episode: tuple) -> float:
        """
        Assuming that a episode is a 4 tuple of the form (s, a, s', r)
        we sum up the rewards (discounted)
        :param episode: (s, a, s', r)
        :return: utility value
        """
        return sum([np.power(self.mdp.gamma, i) * r for i, (_, _, _, r) in enumerate(episode)])


class StationaryPolicy(Policy):

    def __init__(self, actions: dict, mdp: MDP):
        """
        Implements a policy over MDP.
        :param actions: a dict of the form state -> action
        :param mdp: the mdp over which policy is run
        """
        super().__init__(mdp)
        self.actions = actions

    def action(self, state):
        return self.actions[state]



