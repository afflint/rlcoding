import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict
from tqdm import tqdm


class TabularAgent(ABC):
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


class QLearning(TabularAgent):
    def __init__(self, env, gamma = 0.9, learning_rate = 0.01, epsilon = 1, epsilon_decay = 0.05, final_epsilon = 0.1):
        super().__init__(env, gamma, learning_rate, epsilon, epsilon_decay, final_epsilon)
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))

    def V(self, state: Any):
        values = self.Q[state].values()
        if len(values) == 0:
            return 0
        else:
            return max(values)
    
    def greedy(self, state: Any):
        return max(self.Q[state], key=self.Q[state].get)

    def explore(self, state: Any):
        if np.random.uniform() < self.epsilon:
            return self.mdp.action_space.sample()
        else:
            return self.greedy(state=state)
    
    def decay_epsilon(self):
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)
    
    def update(self, state: Any, action: Any, reward: float, terminated: bool, s_prime: Any):
        Q_hat = (not terminated) * self.V(s_prime)
        TD = reward + self.gamma * Q_hat - self.Q[state][action]
        self.Q[state][action] += self.eta * TD
        return TD
    
    def train(self, max_iterations: int = 100000, save_every: int = None):
        if save_every is None:
            save_every = max_iterations
        self.e_decay = self.epsilon / (max_iterations / 2)
        run = list(enumerate(range(max_iterations)))
        for i, episode in tqdm(run):
            state, info = self.mdp.reset()
            done = False
            avg_error = []
            while not done:
                action = self.explore(state)
                s_prime, reward, terminated, truncated, info = self.mdp.step(action)
                error = self.update(state=state, action=action, reward=reward, terminated=terminated, s_prime=s_prime)
                avg_error.append(error)
                done = terminated or truncated 
                state = s_prime
            if i % save_every == 0:
                self.history.append(self.Q.copy())
                self.error.append(np.array(avg_error).min())
            self.decay_epsilon()


class Sarsa(TabularAgent):
    def __init__(self, env, gamma = 0.9, learning_rate = 0.01, epsilon = 1, epsilon_decay = 0.05, final_epsilon = 0.1):
        super().__init__(env, gamma, learning_rate, epsilon, epsilon_decay, final_epsilon)
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))

    def V(self, state: Any):
        values = self.Q[state].values()
        if len(values) == 0:
            return 0
        else:
            return max(values)
    
    def greedy(self, state: Any):
        if len(self.Q[state].values()) > 0:
            return max(self.Q[state], key=self.Q[state].get)
        else:
            return self.mdp.action_space.sample()

    def explore(self, state: Any):
        if np.random.uniform() < self.epsilon:
            return self.mdp.action_space.sample()
        else:
            return self.greedy(state=state)
    
    def decay_epsilon(self):
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)
    
    def update(self, state: Any, action: Any, reward: float, terminated: bool, s_prime: Any):
        a_prime = self.greedy(s_prime)
        Q_hat = (not terminated) * self.Q[s_prime][a_prime]
        TD = reward + self.gamma * Q_hat - self.Q[state][action]
        self.Q[state][action] += self.eta * TD
        return TD
    
    def train(self, max_iterations: int = 100000, save_every: int = None):
        if save_every is None:
            save_every = max_iterations
        self.e_decay = self.epsilon / (max_iterations / 2)
        run = list(enumerate(range(max_iterations)))
        for i, episode in tqdm(run):
            state, info = self.mdp.reset()
            done = False
            while not done:
                action = self.explore(state)
                s_prime, reward, terminated, truncated, info = self.mdp.step(action)
                error = self.update(state=state, action=action, reward=reward, terminated=terminated, s_prime=s_prime)
                done = terminated or truncated 
                state = s_prime
            if i % save_every == 0:
                self.history.append(self.Q.copy())
                self.error.append(error)
            self.decay_epsilon()


class MonteCarlo(TabularAgent):
    def __init__(self, env, gamma = 0.9, learning_rate = 0.01, epsilon = 1, epsilon_decay = 0.05, final_epsilon = 0.1):
        super().__init__(env, gamma, learning_rate, epsilon, epsilon_decay, final_epsilon)
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))
        self.G_values = defaultdict(lambda: 0)
        self.G_count = defaultdict(lambda: 0)

    def V(self, state: Any):
        values = self.Q[state].values()
        if len(values) == 0:
            return 0
        else:
            return max(values)
    
    def greedy(self, state: Any):
        if len(self.Q[state].values()) > 0:
            return max(self.Q[state], key=self.Q[state].get)
        else:
            return self.mdp.action_space.sample()
        
    def generate_episode(self):
        episode = []
        state, _ = self.mdp.reset()
        done = False
        while True:
            action = self.explore(state=state)
            s_prime, reward, done, truncated, _ = self.mdp.step(action=action)
            episode.append((s_prime, int(action), reward))
            if done or truncated:
                break 
            state = s_prime
        return episode

    def explore(self, state: Any):
        if np.random.uniform() < self.epsilon:
            return self.mdp.action_space.sample()
        else:
            return self.greedy(state=state)
    
    def decay_epsilon(self):
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)
    
    def update(self, state: Any, action: Any, reward: float, terminated: bool, s_prime: Any):
        previous_value = self.Q[state][action]
        self.Q[state][action] = self.G_values[(state, action)] / self.G_count[(state, action)]
        TD = self.Q[state][action] - previous_value
        return TD
    
    def train(self, max_iterations: int = 100000, save_every: int = None):
        if save_every is None:
            save_every = max_iterations
        self.e_decay = self.epsilon / (max_iterations / 2)
        run = list(enumerate(range(max_iterations)))

        for i, _ in tqdm(run):
            state, info = self.mdp.reset()
            episode = self.generate_episode()
            visited_pairs = set([(state, action) for state, action, reward in episode])
            for state, action in visited_pairs:
                first_occurrence = next(i for i, x in enumerate(episode) if x[0] == state and x[1] == action)
                G = sum([reward * (self.gamma ** i) for i, (_, _, reward) in enumerate(episode[first_occurrence:])])
                self.G_values[(state, action)] += G 
                self.G_count[(state, action)] += 1
                error = self.update(state, action, None, None, None)
            if i % save_every == 0:
                self.history.append(self.Q.copy())
                self.error.append(error)
            self.decay_epsilon()
