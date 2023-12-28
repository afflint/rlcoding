from typing import Any
import numpy as np
from .utils import kahansum, ATOL


class MarkovChain:
    """
    Markov Chain
    state: binary array where len in the number of states (1 being the current state)
    transition: a |states| x |states| symmetric matrix
    - you can only set up transition states column-wise
    - raise exception when transitions do not sum up to 1
    """

    def __init__(self, number_of_states: int, initial_state: int = 0) -> None:
        self._transition = np.zeros((number_of_states, number_of_states))
        np.fill_diagonal(self._transition, 1)
        self._current_state = np.zeros(number_of_states)
        self._current_state[initial_state] = 1
        self._history = [initial_state]

    def __repr__(self) -> str:
        return self._transition.__repr__()
    
    def __str__(self) -> str:
        return str(self._transition)
    
    def set_transition(self, state: int, transitions: np.ndarray):
        if transitions.ndim == 1 and len(transitions) == self._transition.shape[0] and abs(kahansum(transitions) - 1.) < ATOL:
            self._transition[:,state] = transitions
        else:
            raise ValueError('Transition must be one dimensional array with same len of states and summing up to 1')
    
    @property
    def p_s(self):
        return self.current_state.dot(self._transition.T)
    
    @property
    def current_state(self):
        return self._current_state
    
    @property
    def history(self):
        return self._history
    
    def change_state(self, new_state: int):
        # delete old state
        self._current_state[self._history[-1]] = 0
        # update state
        self._current_state[new_state] = 1
        self._history.append(new_state)
    
    def step(self):
        """
        Starts from current state
        Randomly select a new state
        Move there
        """
        new_state = np.random.choice(list(range(len(self.current_state))), p=self.p_s)
        self.change_state(new_state=new_state)
    
