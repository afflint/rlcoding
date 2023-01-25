from typing import Set

from models import MDP


class StayQuitMDP(MDP):
    """
    Implements the StayQuit game:
    There are two states: {in, end}
    From in you can stay or quit
    - if you quit, you end up in end with probability 1 and reward 10 (default)
    - if you stay:
    - you end up in end with probability 1/3 and reward 4 (default)
    - you end up in in with probability 2/3 and reward 4 (default)
    """
    def __init__(self, gamma: float = 1,
                 stay_in_prob: float = 2/3,
                 stay_in_reward: float = 4.,
                 stay_end_reward: float = 4.,
                 quit_reward: float = 10
                 ):
        super().__init__(gamma)
        self.probabilities = {
            ('IN', 'stay'): {
                'IN': stay_in_prob,
                'END': 1 - stay_in_prob
            },
            ('IN', 'quit'): {'END': 1., 'IN': 0.}
        }
        self.rewards = {
            ('IN', 'stay'): {
                'IN': stay_in_reward,
                'END': stay_end_reward
            },
            ('IN', 'quit'): {'END': quit_reward, 'IN': 0.}
        }

    def start(self):
        return {'IN'}

    def states(self) -> Set:
        return {'IN', 'END'}

    def actions(self, state) -> Set:
        if self.end(state):
            return set()
        else:
            return {'stay', 'quit'}

    def transition(self, start_state, action, end_state) -> float:
        if self.end(start_state):
            return np.nan
        else:
            return self.probabilities[(start_state, action)][end_state]

    def reward(self, start_state, action, end_state) -> float:
        if self.end(start_state):
            return np.nan
        else:
            return self.rewards[(start_state, action)][end_state]

    def end(self, state) -> bool:
        return state == 'END'
