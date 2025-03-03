import numpy as np
from typing import Any
import gymnasium as gym
from tqdm import tqdm
import copy
import torch


class LinearApproximator:
    def __init__(self, feature_extractor: callable, num_features: int, alpha: float=0.01):
        self.feature_extractor = feature_extractor
        self.weights = np.zeros(num_features)
        self.alpha = alpha
    
    def predict(self, state: Any):
        features = self.feature_extractor(state)
        return np.dot(self.weights, features)
    
    def update(self, state: Any, target: float):
        features = self.feature_extractor(state)
        error = target - np.dot(self.weights, features)
        self.weights += self.alpha * error * features


def egreedy_policy(env: gym.Env, approximator: LinearApproximator, state: Any, epsilon: float):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax([approximator.predict(state) for _ in range(env.action_space.n)])

def monte_carlo(env: gym.Env, approximator: LinearApproximator, 
                episodes: int = 1000, gamma: float = 0.99, epsilon: float = 1.0, 
                epsilon_decay: float = 0.995, min_epsilon: float = 0.1):
    for _ in tqdm(range(episodes), total=episodes):
        state, _ = env.reset()
        trajectory = []
        done = False
        while not done:
            action = egreedy_policy(env, approximator, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append((state, reward))
            state = next_state
        # Returns
        G = 0
        for state, reward in reversed(trajectory):
            G = reward + gamma * G
            approximator.update(state=state, target=G)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

def td_0(env: gym.Env, approximator: LinearApproximator, 
         episodes: int = 1000, gamma: float = 0.99, 
         epsilon: float = 1.0, epsilon_decay: float = 0.995, min_epsilon: float = 0.1):
    for _ in tqdm(range(episodes), total=episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = egreedy_policy(env, approximator, state, epsilon)
            # TD(0) target
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if not done:
                target = reward + (gamma * approximator.predict(next_state))
                approximator.update(state, target)
                state = next_state
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

class LinearQApproximator:
    def __init__(self, feature_extractor: callable, num_features: int, num_actions: int, alpha=0.01):
        self.feature_extractor = feature_extractor
        self.weights = np.ones((num_actions, num_features))
        self.alpha = alpha
        self.error_log = []
    
    def predict(self, state, action):
        features = self.feature_extractor(state)
        return np.dot(self.weights[action], features)
    
    def update(self, state, action, target):
        features = self.feature_extractor(state)
        error = target - np.dot(self.weights[action], features)
        #self.error_log.append(error)
        self.weights[action] += self.alpha * error * features

def egreedy_policy_q(env: gym.Env, approximator: LinearApproximator, state: Any, epsilon: float):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax([approximator.predict(state, action) for action in range(env.action_space.n)])
    
def sarsa(env: gym.Env, approximator: LinearQApproximator, 
          episodes: int = 1000, gamma: float = 0.99, 
          epsilon: float = 1.0, epsilon_decay: float = 0.995, min_epsilon: float = 0.1, 
          keep_history: bool = False):
    history = []
    for _ in tqdm(range(episodes), total=episodes):
        state, _ = env.reset()
        action = egreedy_policy_q(env, approximator, state, epsilon)
        done = False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated:
                target = reward
            elif truncated:
                target = approximator.predict(state, action)
            else:
                next_action = egreedy_policy_q(env, approximator, next_state, epsilon)
                target = reward + (gamma * approximator.predict(next_state, next_action))
                approximator.update(state, action, target)
                state, action = next_state, next_action

        if keep_history:
            history.append(copy.deepcopy(approximator.weights))
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    return history

def q_learning(env: gym.Env, approximator: LinearQApproximator, 
          episodes: int = 1000, gamma: float = 0.99, 
          epsilon: float = 1.0, epsilon_decay: float = 0.995, min_epsilon: float = 0.1, 
          keep_history: bool = False):
    history = []
    for _ in tqdm(range(episodes), total=episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = egreedy_policy_q(env, approximator, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated:
                target = reward
            elif truncated:
                target = approximator.predict(state, action)
            else:
                best_next_action = np.argmax([approximator.predict(next_state, a) for a in range(env.action_space.n)])                
                target = reward + (gamma * approximator.predict(next_state, best_next_action))
                approximator.update(state, action, target)
                state = next_state

        if keep_history:
            history.append(copy.deepcopy(approximator.weights))
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    return history
