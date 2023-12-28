from abc import ABC, abstractmethod, abstractproperty
from typing import Set, List, Dict, Tuple
import numpy as np


class MDP(ABC):
    """
    Implements a Markov Decision Process that is organized as follows
    States and Actions are indexed by numerical values
    - states() -> Set[int]: Set of all the possible states
    - actions() -> Set[int]: Set of all the possible actions
    - gamma (float): Discount factor 0 < gamma <= 1
    - probability(s, a, s') -> float: Transition probability to state s' from state s choosing action a
    - reward(s, a, s') -> float: Reward in state s' from state s choosing action a
    - is_terminal(s) -> bool: True if s is a terminal state 
    - transition(s, a, s') -> Tuple[float, float]: Probability and Reward utility function
    - options(s, a) -> Dict[int, Tuple[float, float]]: Dictionary of non-zero probability states s'

    Args:
        ABC (class): This is an abstract class
    """

    def __init__(self, gamma: float = .9):
        self.gamma = gamma

    @abstractmethod
    def start(self) -> int:
        """
        Init the starting state

        Returns:
            int: Starting state
        """
        pass
    
    @abstractmethod
    def is_terminal(self, s: int) -> bool:
        """
        Check if state s is the terminal state

        Args:
            s (int): State
        
        Returns:
            bool: True if s is terminal
        """
    
    @abstractproperty
    def states(self) -> Set[int]:
        """
        Get the possible states

        Returns:
            Set[int]: Set of all the possible states
        """
        pass
        
    @abstractproperty
    def actions(self) -> Set[int]:
        """
        Get all the possible actions

        Returns:
            Set[int]: Set of all the possible actions
        """
        pass
    
    @abstractmethod
    def probability(self, s: int, a: int, s_prime: int) -> float:
        """
        Probability of s_prime from state s choosing action a

        Args:
            s (int): Start state
            a (int): Action chosen
            s_prime (int): Ending state

        Returns:
            float: Probability p (i.e., 0 <= p <= 1)
        """
        pass
    
    @abstractmethod
    def reward(self, s: int, a: int, s_prime: int) -> float:
        """
        Reward of s_prime from state s choosing action a

        Args:
            s (int): Start state
            a (int): Action chosen
            s_prime (int): Ending state

        Returns:
            float: Reward
        """
        
    def transition(self, s: int, a: int, s_prime: int) -> Tuple[float, float]:
        """
        Utility for Probability + Reward

        Args:
            s (int): Start state
            a (int): Action chosen
            s_prime (int): Ending state

        Returns:
            Tuple[float, float]: Probability and Reward
        """
        return self.probability(s, a, s_prime), self.reward(s, a, s_prime)
    
    def options(self, s: int, a: int) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Utility for getting the probability and reward for all the 
        non-zero probability ending states

        Args:
            s (int): Starting state
            a (int): Action

        Returns:
            List[Tuple[int, Tuple[float, float]]]: List of Probability and Reward per State prime (only non-zero probabilities)
        """
        transitions = []
        for x in self.states:
            p, r = self.transition(s, a, x)
            if p > 0:
                transitions.append((x, (p, r)))
        return transitions

    
        
    
