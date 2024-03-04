"""Agents for RL"""
import numpy as np 
import gymnasium as gym
import torch
from abc import abstractmethod
from collections import defaultdict
from typing import Any
from tqdm import tqdm


class Agent:
    """Generic Agent for Tabular Learning
    """
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
        
    def greedy_policy(self, state: Any) -> int:
        """Provides a greedy policy

        Args:
            state (Any): Observed state

        Returns:
            int: action
        """
        return int(np.argmax(self.Q[state]))
    
    def e_greedy_policy(self, state: Any) -> int:
        """Provides epsilon-greedy policy

        Args:
            state (Any): The observed state

        Returns:
            int: action
        """
        if np.random.random() < self.epsilon:
            a = self.env.action_space.sample()
        else:
            a = int(np.argmax(self.Q[state]))
        return a
    
    def stochastic_policy(self, state: Any) -> int:
        """Provides a sochastic policy

        Args:
            state (Any): _description_

        Returns:
            int: _actiin
        """
        p = np.exp(self.Q[state]) / sum(self.Q[state])
        return np.random.choice(list(range(0, self.env.action_space.n)), p=p)
    
    def decay_epsilon(self):
        """Reduces epsilon during training
        """
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)

    @abstractmethod
    def update(self, state, action, reward, terminated, s_prime):
        pass 
            

class QLearningAgent(Agent):
    """Agent that implements Q-learning as method update
    """
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

    
class AgentVFA:
    """Generic Agent for Value Function Approxiation with discrete actions"""
    def __init__(self, n_features: int, environment: gym.Env, learning_rate: float, epsilon: float, 
                epsilon_decay: float,
                final_epsilon: float, 
                gamma: float = .95) -> None:
        self.size = n_features
        self.env = environment
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = epsilon_decay
        self.e_final = final_epsilon
        self.training_error = []
        w_i = np.ones(self.size)
        self.w = torch.tensor(w_i, dtype=float, requires_grad=True)
    
    @abstractmethod
    def x(self, state: Any) -> torch.Tensor:
        """Returns the features observing the state

        Args:
            state (Any): Any Gym state

        Returns:
            torch.Tensor: Feature representation of the array
        """
    
    def v(self, state: Any) -> torch.Tensor:
        """Approximated Value Function

        Args:
            state (Any): Any Gym State

        Returns:
            torch.Tensor: State Value
        """
        return self.x(state) @ self.w
    
    def greedy_policy(self, state: Any) -> int:
        """Gets the action leading to the best state

        Args:
            state (Any): Current state

        Returns:
            int: action chosen
        """
        values = []
        for candidate_action in range(self.env.action_space.n):
            s_prime, _, _, _, _ = self.env.step(action=candidate_action)
            values.append(self.v(s_prime).detach().numpy())
        return np.argmax(values)
    
    def e_greedy_policy(self, state: Any) -> int:
        """Provides epsilon-greedy policy

        Args:
            state (Any): The observed state

        Returns:
            int: action
        """
        if np.random.random() < self.epsilon:
            a = self.env.action_space.sample()
        else:
            a = self.greedy_policy(state)
        return a


    def td_learning(self, max_iterations: int = 1_000):
        """TD(0) learning with e-greedy policy

        Args:
            max_iterations (int, optional): Maximum number of iterations. Defaults to 1_000.
        """
        state, _ = self.env.reset()
        for i in tqdm(range(max_iterations)):
            action = self.e_greedy_policy(state)
            s_prime, reward, terminated, truncated, _ = self.env.step(action=action)
            target = reward + self.gamma * self.v(s_prime)
            v_hat = self.v(state)
            delta = target - v_hat
            v_hat.backward()
            with torch.no_grad():
                self.w += self.alpha * delta * self.w.grad 
                self.w.grad.zero_()
            self.training_error.append(delta.detach().numpy())
            if terminated or truncated:
                self.decay_epsilon()
                state, _ = self.env.reset()
            else:
                state = s_prime
    
    def decay_epsilon(self):
        """Reduce epsilon for e-greedy policies"""
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)