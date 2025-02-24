import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict
from tqdm import tqdm


class LinearVFA(ABC):
    def __init__(self, env: gym.Env, gamma: float = 0.9, 
                learning_rate: float = .01, 
                epsilon: float = 1,
                epsilon_decay: float = .05,
                final_epsilon: float = 0):
        self.mdp = env 
        self.gamma = gamma 
        self.eta = learning_rate
        self.epsilon = epsilon
        self.e_decay = epsilon_decay
        self.e_final = final_epsilon
        self.history = []
        self.error = []

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def explore(self):
        pass

    @abstractmethod
    def train(self, max_iterations: int = 100_000):
        pass