from typing import Any, Generator, List, Tuple
from collections import namedtuple

import numpy as np
import pandas as pd
import mdp.policy as ply


Transition = namedtuple('Transition', ['s_start', 'action', 's_prime', 'probability', 'reward'])

class MDP:
    """
    Models a Markov Decision Process featured by:
    - a finite set S of states
    - a finite set A of actions
    - a transition tensor T = S x A x S' where sum(Tsa) = 1 is the distribution of probabilities
    of a transition from s and a to any s'
    - a reward tensor R = S x A x S' (same as T but giving the reward)
    """
    def __init__(self, states_number: int, actions_number: int, config: List[Transition] = None, gamma: float = .9) -> None:
        """
        Init the MDP according to the configuration

        Args:
            config (List[dict]): (s_start, action, s_prime, probability, reward). Transitions not specified have a 
            default probability of 0 and reward of 0
            gamma (float, optional): discount factor. Defaults to .9.
        """
        self.states = dict((s, s) for s in range(states_number))
        self.actions = dict((s, s) for s in range(actions_number))
        self.state_id = dict((n, i) for i, n in self.states.items())
        self.action_id = dict((n, i) for i, n in self.actions.items())
        self.T = np.zeros((states_number, actions_number, states_number))
        self.R = np.zeros((states_number, actions_number, states_number))
        self.gamma = gamma
        if config is not None:
            for t in config:
                self.T[t.s_start, t.action, t.s_prime] = t.probability
                self.R[t.s_start, t.action, t.s_prime] = t.reward
    
    def set_state_names(self, **kwargs):
        """
        Provides name=index for states
        """
        for name, idx in kwargs.items():
            self.states[idx] = name
            try:
                del self.state_id[idx]
            except KeyError:
                pass
            self.state_id[name] = idx
    
    def set_action_names(self, **kwargs):
        """
        Provides name=index for actions
        """
        for name, idx in kwargs.items():
            self.actions[idx] = name
            try:
                del self.action_id[idx]
            except KeyError:
                pass
            self.action_id[name] = idx
    
    def is_terminal(self, state: int):
        return self.T[state, :].sum() == 0
    
    def successors(self, state: int, action: int) -> Generator[tuple[int, float, float], Any, None]:
        """
        Get all the states reachable from state choosing action

        Args:
            state (int): state we are moving from
            action (int): action chosen

        Yields:
            Generator[tuple[int, float, float]]: s_prime, probability, reward
        """
        P, R = self.T[state, action], self.R[state, action]
        for i, p in enumerate(P):
            if p > 0:
                yield i, p, R[i]
    
    def to_table(self) -> pd.DataFrame:
        rows = []
        for state in self.states.keys():
            for action in self.actions.keys():
                for s_prime, p, r in self.successors(state, action):
                    rows.append({
                        'state': self.states[state],
                        'action': self.actions[action],
                        's_prime': self.states[s_prime],
                        'probability': p,
                        'reward': r
                    })
        return pd.DataFrame(rows)          
    
    def step(self, state: int, policy: ply.Policy) -> Tuple[int, int, int, float]:
        """
        Do a step according to a policy. 

        Args:
            state (int): current state
            policy (ply.Policy): policy used to select actions

        Raises:
            ValueError: if the state is terminal we do not have any other action

        Returns:
            Tuple[int, int, int, float]: state, action, s_prime, reward
        """
        if self.is_terminal(state):
            raise ValueError('State {} is the terminal state'.format(state))
        else:
            action = policy[state]
            p = self.T[state, action]
            s_prime = np.random.choice(range(len(self.actions)), p=p) 
            return state, action, s_prime, self.R[state, action, s_prime]
    
    def episode(self, init_state: int, policy: ply.Policy, max_iterations: int = 10000) -> List[Tuple[int, int, int, float]]:
        """
        Generate a full episode for a maximum number of iterations

        Args:
            init_state (int): state to start from
            max_iterations (int, optional): maximum number of steps. Defaults to 10000.

        Returns:
            List[Tuple[int, int, int, float]]: S_0, A_0, R_1, S_1, A_1, R_2, ..., S_t-1, A_t-1, R_t
        """
        episode = []
        current_state = init_state
        for i in range(max_iterations):
            s, a, s_prime, r = self.step(state=current_state, policy=policy)
            episode.append((s, a, r))
            current_state = s_prime
            if self.is_terminal(current_state):
                break
        return episode
