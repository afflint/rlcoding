from typing import Set

import numpy as np

from models import MDP


class DungeonMDP(MDP):
    """
    Represents a dungeon where from each room you can always go LEFT or RIGHT.
    There are 4 states:
    - EMPTY ROOM (reward 0)
    - MONSTER ROOM (reward -X)
    - TREASURE ROOM (reward Y)
    - EXIT (out of game)
    LEFT brings to EMPTY, MONSTER, TREASURE, OUT with p = alpha (array over actions)
    RIGHT brings to EMPTY, MONSTER, TREASURE, OUT with p = beta (array over actions)
    """
    def __init__(self, alpha=None, beta=None, gamma=.8):
        super().__init__(gamma)
        self.schema = ['E', 'M', 'T', 'O']
        self.rewards = [0, -30, 10, 0]
        if alpha is None:
            self.alpha = np.array([.55, .1, .3, .05])
        else:
            self.alpha = alpha
        if beta is None:
            self.beta = np.array([.35, .20, .40, .05])
        else:
            self.beta = beta

    def start(self):
        return {'E'}

    def states(self) -> Set:
        return {'E', 'M', 'T', 'O'}

    def actions(self, state) -> Set:
        return {'R', 'L'}

    def transition(self, start_state, action, end_state) -> float:
        p_index = self.schema.index(end_state)
        p = 0
        if action == 'L':
            p = self.alpha[p_index]
        elif action == 'R':
            p = self.beta[p_index]
        return p

    def reward(self, start_state, action, end_state) -> float:
        p_index = self.schema.index(end_state)
        return self.rewards[p_index]

    def end(self, state) -> bool:
        return state == 'O'

    def successors(self, state, action):
        steps = []
        if not self.end(state):
            for s_prime in self.states():
                steps.append((
                    s_prime,
                    self.transition(state, action, s_prime),
                    self.reward(state, action, s_prime)
                ))
        return steps

