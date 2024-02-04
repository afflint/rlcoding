"""Implements the Forest Agent"""
import numpy as np
import gymnasium as gym
import torch 


class ForestAgentQlearning:
    def __init__(self, environment: gym.Env, learning_rate: float, epsilon: float, 
                epsilon_decay: float,
                final_epsilon: float, 
                gamma: float = .95) -> None:
        self.env = environment
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        for i, row in enumerate(self.env.unwrapped.available_actions):
            for j, a in enumerate(row):
                if a == 0:
                    self.Q[i, j] = -np.inf # set -inf for unavailable actions
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = epsilon_decay
        self.e_final = final_epsilon
        self.training_error = []
        
    def policy(self, state: int) -> int:
        """Implements e-greedy strategy for action selection

        Args:
            state (tuple[int, int, bool]): state observed according to the environment

        Returns:
            int: action
        """
        if np.random.random() < self.epsilon:
            options = [i for i, a in enumerate(self.env.unwrapped.available_actions[state]) if a == 1]
            return np.random.choice(options)
        else:
            return int(np.argmax(self.Q[state]))
    
    def update(self, state: int, action: int, reward: float, terminated: bool, s_prime: int):
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


class ForestAgentQlearningVFA:
    def __init__(self, environment: gym.Env, learning_rate: float, epsilon: float, 
                epsilon_decay: float,
                final_epsilon: float, 
                gamma: float = .95, initial_w: np.ndarray = None) -> None:
        self.env = environment
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = epsilon_decay
        self.e_final = final_epsilon
        self.training_error = []
        if initial_w is None:
            i_w = np.ones((self.env.action_space.n, self.env.action_space.n))
        else:
            i_w = initial_w
        self.w = torch.tensor(i_w, dtype=float, requires_grad=True)
    
    def x(self, state: int) -> np.ndarray:
        """Returns the features that represents the state

        Args:
            state (int): state

        Returns:
            np.ndarray: state perceived by the agent in terms of available directions 
        """
        return torch.tensor(self.env.unwrapped.available_actions[state], dtype=float, requires_grad=False)
    
    def q(self, state: int, action: int) -> float:
        return self.x(state) @ self.w[:, action]
        
    def policy(self, state: int) -> int:
        """Implements e-greedy strategy for action selection

        Args:
            state (tuple[int, int, bool]): state observed according to the environment

        Returns:
            int: action
        """
        available = self.env.unwrapped.available_actions[state]
        options = [i for i, a in enumerate(available) if a == 1]
        if np.random.random() < self.epsilon:
            return np.random.choice(options)
        else:
            available_values = [self.q(state, a).detach().numpy() for a in options]
            return options[np.argmax(available_values)]

    def update(self, state: int, action: int, reward: float, s_prime: int):
        q_target = reward + self.gamma * max(self.q(s_prime, a) for a in self.env.unwrapped.available_actions[s_prime])
        q_value = self.q(state, action)
        delta = q_target - q_value
        q_value.backward()
        with torch.no_grad():
            self.w += self.alpha * delta * self.w.grad 
            self.w.grad.zero_()
        self.training_error.append(delta.detach().numpy())
    
    def decay_epsilon(self):
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)
    