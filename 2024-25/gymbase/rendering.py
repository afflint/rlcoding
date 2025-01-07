import gymnasium as gym

def frozen_lake_exploration():
    env = gym.make('FrozenLake-v1', render_mode="human")
    observation, info = env.reset()

    end_episode = False 
    while not end_episode:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        end_episode = terminated or truncated
    env.close()

frozen_lake_exploration()