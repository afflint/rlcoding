from collections import defaultdict
from typing import Any
import numpy as np 
import gymnasium as gym
from abc import abstractmethod


class Agent:
    def __init__(self,
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
        
    def greedy_policy(self, state: Any):
        return int(np.argmax(self.Q[state]))
    
    def e_greedy_policy(self, state: Any):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.Q[state]))
    
    def stochastic_policy(self, state: Any):
        p = np.exp(self.Q[state]) / sum(self.Q[state])
        return np.random.choice(list(range(0, self.env.action_space.n)), p=p)
    
    def decay_epsilon(self):
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)

    @abstractmethod
    def update(self, state, action, reward, terminated, s_prime):
        pass 
            

class QLearningAgent(Agent):
    def __init__(self, environment: gym.Env, learning_rate: float, epsilon: float, epsilon_decay: float, final_epsilon: float, gamma: float = 0.95) -> None:
        super().__init__(environment, learning_rate, epsilon, epsilon_decay, final_epsilon, gamma)
    
    def update(self, state: Any, action: Any, reward: Any, terminated: bool, s_prime: Any):
        """Q-learning update rule

        Args:
            state (Any): agent state
            action (Any): action chosen
            reward (Any): reward received
            terminated (bool): check if terminal state
            s_prime (Any): detination state
        """
       # The line `Q_hat = (not terminated) * np.max(self.Q[s_prime])` is calculating the estimated
       # maximum future reward for the next state `s_prime`.
        Q_hat = (not terminated) * np.max(self.Q[s_prime])
        TD = (reward + self.gamma * Q_hat - self.Q[state][action])
        # update
        self.Q[state][action] += self.alpha * TD
        self.training_error.append(TD)

    