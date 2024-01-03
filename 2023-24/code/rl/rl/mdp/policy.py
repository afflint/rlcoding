from typing import Tuple, List
from rl.mdp.model import MDP
from abc import abstractmethod
import numpy as np


class Policy:
    def __init__(self, mdp: MDP) -> None:
        self.mdp = mdp
    
    @abstractmethod
    def __getitem__(self, state: int) -> int:
        """
        Get action

        Args:
            state (int): State

        Returns:
            int: The action chosen
        """
        pass
    
    def step(self, state: int) -> Tuple[int, int, int, float]:
        """
        Take a step according to the policy

        Args:
            state (int): State

        Returns:
            Tuple[int, int, int, float]: starting state, action chosen, ending state, reward
        """
        action = self[state]
        options = self.mdp.options(s=state, a=action)
        try:
            outcome = np.random.choice(range(len(options)), p=[x[1][0] for x in options])
        except ValueError:
            print(state, options)
        s_prime, (_, r) = options[outcome]
        return (state, action, s_prime, r)
    
    def episode(self, max_iterations: int = 10000) -> List[Tuple[int, int, int, float]]:
        """
        Generate an episode according to the Policy

        Returns:
            List[Tuple[int, int, int, float]]: S_0, A_0, R_1, S_1, A_1, R_2, ..., S_t-1, A_t-1, R_t 
        """
        episode = []
        current_state = self.mdp.start()
        for i in range(max_iterations):
            s, a, s_prime, r = self.step(state=current_state)
            episode.append((s, a, r))
            current_state = s_prime
            if self.mdp.is_terminal(current_state):
                break
        return episode
        
