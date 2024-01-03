import gymnasium as gym
import matplotlib.pyplot as plt 

env = gym.make("PongNoFrameskip-v4", render_mode="human")
observation, info = env.reset()
score, history = 0, []
for _ in range(100):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    score += reward
    history.append((observation, reward, info))
    if terminated or truncated:
        observation, info = env.reset()
env.close()

print(f"Game score is {score}")

print(history[0][0])
print(history[0][0].shape)
print(history[0][1])
print(history[0][2])
print(env.action_space)

fig, ax = plt.subplots()
ax.imshow(history[0][0], cmap="gray")
plt.tight_layout()
plt.show()