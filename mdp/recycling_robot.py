from typing import Set

import models


class RecyclingMDP(models.MDP):

    def __init__(self, alpha: float, beta: float,
                 r_search: float, r_wait: float, gamma: float=.99):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.r_search = r_search
        self.r_wait = r_wait

        self.probs = {
            ('H', 'wait', 'H'): 1,
            ('H', 'wait', 'L'): 0,
            ('H', 'search', 'H'): self.alpha,
            ('H', 'search', 'L'): 1 - self.alpha,
            ('L', 'wait', 'H'): 0,
            ('L', 'wait', 'L'): 1,
            ('L', 'recharge', 'H'): 1,
            ('L', 'recharge', 'L'): 0,
            ('L', 'search', 'H'): 1 - self.beta,
            ('L', 'search', 'L'): self.beta,
        }

        self.rews = {
            ('H', 'wait', 'H'): self.r_wait,
            ('H', 'wait', 'L'): self.r_wait,
            ('H', 'search', 'H'): self.r_search,
            ('H', 'search', 'L'): self.r_search,
            ('L', 'wait', 'H'): self.r_wait,
            ('L', 'wait', 'L'): self.r_wait,
            ('L', 'recharge', 'H'): 0,
            ('L', 'recharge', 'L'): 0,
            ('L', 'search', 'H'): -3,
            ('L', 'search', 'L'): self.r_search,
        }

    def start(self):
        return {'H'}

    def states(self) -> Set:
        return {'H', 'L'}

    def actions(self, state) -> Set:
        if state == 'H':
            return {'search', 'wait'}
        else:
            return {'search', 'wait', 'recharge'}

    def transition(self, start_state, action, end_state) -> float:
        try:
            return self.probs[(start_state, action, end_state)]
        except KeyError:
            return 0.

    def reward(self, start_state, action, end_state) -> float:
        try:
            return self.rews[(start_state, action, end_state)]
        except KeyError:
            return 0.

    def end(self, state) -> bool:
        return False