from typing import Set

from rl.mdp.model import MDP


class RandomWalk(MDP):
    """
    Implements example 6.1 in Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.

    Args:
        MDP (class): Abstract MDP
    """
    
    def __init__(self, steps: int = 10, gamma: float = 0.9):
        super().__init__(gamma)
        self.number_of_steps = steps
    
    def start(self) -> int:
        return self.number_of_steps // 2
    
    def is_terminal(self, s: int) -> bool:
        return s == 1 or s == self.number_of_steps
    
    @property
    def states(self) -> Set[int]:
        return set(range(1, self.number_of_steps + 1))
    
    @property
    def actions(self) -> Set[int]:
        return {0}
    
    def probability(self, s: int, a: int, s_prime: int) -> float:
        if s_prime == s + 1:
            return .5
        elif s_prime == s - 1: 
            return .5
        else:
            return 0   
        
    def reward(self, s: int, a: int, s_prime: int) -> float:
        if s_prime == self.number_of_steps:
            return 1
        else:
            return 0
        