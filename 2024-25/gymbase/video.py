import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/7.1_4/bin/ffmpeg"

env = gym.make('MountainCar-v0', render_mode="rgb_array")
env = RecordVideo(env=env, video_folder="./2024-25/data", 
                  episode_trigger=lambda x: x % 2 == 0)
env.reset()

env.start_recording(video_name="video")
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated: break
env.stop_recording()
env.close()