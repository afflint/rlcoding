from collections import defaultdict
import numpy as np 
import gymnasium as gym

class BlackJackAgent:
    def __init__(
        self, 
        environment: gym.Env,
        learning_rate: float, 
        epsilon: float, 
        epsilon_decay: float, 
        final_epsilon: float, 
        gamma: float = .95) -> None:
        self.env = environment
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = epsilon_decay
        self.e_final = final_epsilon
        self.training_error = []
        
    def policy(self, state: tuple[int, int, bool]) -> int:
        """Implements e-greedy strategy for action selection

        Args:
            state (tuple[int, int, bool]): state observed according to the environment

        Returns:
            int: action
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.Q[state]))
    
    def update(self, state: tuple[int, int, bool], action: int, reward: float, terminated: bool, s_prime: tuple[int, int, bool]):
        """Q-learning update rule

        Args:
            state (tuple[int, int, bool]): s
            action (int): a
            reward (float): r
            terminated (bool): if final
            s_prime (tuple[int, int, bool]): s'
        """
        Q_hat = (not terminated) * np.max(self.Q[s_prime])
        TD = (reward + self.gamma * Q_hat - self.Q[state][action])
        # update
        self.Q[state][action] += self.alpha * TD
        self.training_error.append(TD)
    
    def decay_epsilon(self):
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)
    