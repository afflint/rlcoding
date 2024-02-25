"""Module providing an example for VFA modeling

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
import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


sns.set_theme(style="whitegrid")


variables = ['Pos X', 'Pos Y', 'Vel X', 'Vel Y', 'Angle', 'Ang Vel', 'LonG', 'RonG']
actions = {
    0: "Do nothing", 1: "Left engine", 2: "Main engine", 3: "Right engine"
}

env = gym.make(
    "LunarLander-v2",
    continuous = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,
    render_mode="human"
)

def print_observation(observation):
    """Print observations

    Args:
        observation (nd.array): state
    """
    for i, f in enumerate(observation):
        print(f"{variables[i]:10}: {f:0.2f}")

def plot(observation_data):
    """Plots observation variables

    Args:
        observation_data (pandas dataframe): contains history of episode
    """
    palette = sns.color_palette("mako_r", 6)
    _, ax = plt.subplots(figsize=(12, 8), ncols=3, nrows=3)
    A = np.array(variables + ['Reward']).reshape(3, 3)
    for x, row in enumerate(A):
        for y, a in enumerate(row):
            sns.lineplot(data=observation_data, y=a, 
                        x=range(observation_data.shape[0]),
                        markers=True, ax=ax[x, y])
            ax[x, y].set_title(a)
    plt.tight_layout()
    plt.show()

def main():
    """Execute script
    """
    observation, _ = env.reset()
    print_observation(observation=observation)
    collected_data = [list(observation) + [None, 0]]
    for i in range(1000):
        print("===============")
        print(f"INTERACTION {i+1}")
        print("===============")
        action = env.action_space.sample()
        print(f"Do action {actions[action]}")
        print("===============")
        new_observation, reward, stop, _, _ = env.step(action=0)
        collected_data.append(list(new_observation) + [actions[action], reward])
        print(f"Reward {reward}")
        print("===============")
        print_observation(new_observation)
        time.sleep(.0002)
        if stop:
            break
    history = pd.DataFrame(np.array(collected_data), columns=variables + ['Action', 'Reward'])
    return history

if __name__ == '__main__':
    data =  main()
    plot(data)
