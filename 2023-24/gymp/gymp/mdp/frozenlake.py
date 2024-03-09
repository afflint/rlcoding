import gymnasium as gym
from tqdm import tqdm 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import torch
from agents import AgentQFA

sns.set_theme(style="whitegrid")

class QLearningAgent(AgentQFA):
    def x(self, state, action):
        # one hot encoded actions
        av = [1 if a == action else 0 for a in range(self.env.action_space.n)]
        # one hot encoded states
        st = [0]*self.env.observation_space.n
        st[state] = 1
        f = np.array(st + av)
        return torch.tensor(f, dtype=float, requires_grad=False)


## Initialize
env = gym.make("FrozenLake-v1", render_mode=None)
learning_rate = 0.1
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.05

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

## Qlearning
agent = QLearningAgent(
    n_features=env.observation_space.n + env.action_space.n, environment=env, learning_rate=learning_rate, epsilon=start_epsilon, 
    epsilon_decay=epsilon_decay, final_epsilon=final_epsilon, gamma=.95
)

## Execute episode without training
testenv = gym.make("FrozenLake-v1", render_mode="human")
testenv.metadata['render_fps'] = 24
for episode in tqdm(range(5)):
    state, info = testenv.reset()
    done = False
    # episode run
    while not done:
        action = testenv.action_space.sample()
        s_prime, reward, terminated, truncated, info = testenv.step(action=action)
        done = terminated or truncated
        state = s_prime

## Training
agent.q_learning(max_iterations=n_episodes)
        
## Execute episode
testenv2 = gym.make("FrozenLake-v1", render_mode="human")
testenv2.metadata['render_fps'] = 12
for episode in tqdm(range(10)):
    state, info = testenv2.reset()
    done = False
    # episode run
    while not done:
        action = agent.e_greedy_policy(state)
        s_prime, reward, terminated, truncated, info = testenv2.step(action=action)
        done = terminated or truncated
        state = s_prime


qa = np.zeros((env.observation_space.n, env.action_space.n))
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        qa[s, a] = agent.q(s, a)
        
fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(qa, ax=ax, annot=True, fmt=".3e", cmap='crest')
plt.tight_layout()
plt.show()
