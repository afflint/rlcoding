"""Training an agent for Lunar Landing

Action space
    0: do nothing
    1: fire left orientation engine
    2: fire main engine
    3: fire right orientation engine
    
For each step, the reward:
- is increased/decreased the closer/further the lander is to the landing pad
- is increased/decreased the slower/faster the lander is moving.
- is decreased the more the lander is tilted (angle not horizontal).
- is increased by 10 points for each leg that is in contact with the ground.
- is decreased by 0.03 points each frame a side engine is firing.
- is decreased by 0.3 points each frame the main engine is firing.
The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.
    
"""

import time
import vfa
import torch
import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


sns.set_theme(style="whitegrid")


def create_environment(render_mode):
    """Creates environment with render mode"""
    env = gym.make(
        "LunarLander-v2",
        continuous = False, 
        gravity = -10.0,
        enable_wind = False,
        wind_power = 15.0,
        turbulence_power = 1.5,
        render_mode=render_mode)
    return env

def greedy(env, state, w):
    """Implements the greedy policy"""
    def q(state, action):
        x = torch.tensor(state, dtype=float, requires_grad=False)
        return x @ w[:, action]
    a = int(np.argmax(np.array([q(state, action).detach().numpy() for action in range(env.action_space.n)])))
    return a
    
def main():
    """Main execution
    1. Run 10 episodes with no training
    2. Train by SARSA
    3. Run 10 episodes with trained agent
    """
    # Run with no training
    env = create_environment(render_mode="human")
    for _ in range(10):
        _, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            _, _, stop, truncated, _ = env.step(action=action)
            done = stop or truncated
            if done:
                break
    # Train
    env = create_environment(render_mode="rgb_array")
    state, _ = env.reset()
    w, history, rewards = vfa.sarsa(env=env, n_steps=10_000, n_features=8)
    
    # Run some episodes with the greedy policy
    env = create_environment(render_mode="human")
    for _ in range(10):
        state, _ = env.reset()
        done = False
        while not done:
            action = greedy(env=env, state=state, w=w)
            state, _, stop, truncated, _ = env.step(action=action)
            done = stop or truncated
            if done:
                break
    return history, rewards

def conv(a, win=100):
    return np.convolve(a, np.ones(win), mode='same') / win
    
if __name__ == '__main__':
    history, rewards = main()
    r = conv(rewards)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x=range(len(r)), y=r)
    plt.tight_layout()
    plt.show()
