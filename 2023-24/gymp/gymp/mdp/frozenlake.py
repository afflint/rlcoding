from agents import QLearningAgent
import gymnasium as gym
from tqdm import tqdm 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
sns.set_theme(style="whitegrid")


## Initialize
env = gym.make("FrozenLake-v1", render_mode=None)
learning_rate = 0.001
n_episodes = 10_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

## Qlearning
agent = QLearningAgent(
    environment=env, learning_rate=learning_rate, epsilon=start_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon, gamma=.95
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
for episode in tqdm(range(n_episodes)):
    state, info = env.reset()
    done = False
    # episode run
    while not done:
        action = agent.e_greedy_policy(state)
        s_prime, reward, terminated, truncated, info = env.step(action=action)
        # update
        agent.update(state, action, reward, terminated, s_prime)
        done = terminated or truncated
        state = s_prime
    agent.decay_epsilon()
        
## Execute episode
testenv2 = gym.make("FrozenLake-v1", render_mode="human")
testenv2.metadata['render_fps'] = 12
for episode in tqdm(range(10)):
    state, info = testenv2.reset()
    done = False
    # episode run
    while not done:
        action = agent.greedy_policy(state)
        s_prime, reward, terminated, truncated, info = testenv2.step(action=action)
        done = terminated or truncated
        state = s_prime


space = np.zeros(len(agent.Q))
for s, values in agent.Q.items():
    v = values.max()
    print(space[s])
    space[s] = v 
    print(space[s])
V = space.reshape(int(np.sqrt(len(agent.Q))), -1)

fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(V, ax=ax, annot=True, fmt=".3e", cmap='crest')
plt.tight_layout()
plt.show()

print(pd.DataFrame(V))