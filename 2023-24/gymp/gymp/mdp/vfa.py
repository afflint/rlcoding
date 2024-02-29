"""Collection of VFA Algorithms"""
import torch
from tqdm import tqdm
import numpy as np
import gymnasium as gym


def qlearning(env: gym.Env, n_steps: int, n_features: int, 
        x: callable = None, gamma = .99, alpha = 0.01, 
        initial_w = None, epsilon: float = 0.5, final_epsilon: float = 0.1):
    """Off policy Q-Learning"""
    w = torch.tensor(np.ones((n_features, env.action_space.n)), dtype=float, requires_grad=True)
    x = lambda state: torch.tensor(state, dtype=float, requires_grad=False)
    
    def Q(state, action):
        return x(state) @ w[:, action]
    
    # train
    state, _ = env.reset()
    for i in range(n_steps):
        action = env.action_space.sample()
        s_prime, reward, done, truncated, _ = env.step(action)
        mx = max(Q(s_prime, a) for a in range(env.action_space.n))
        delta = alpha * (reward + gamma * mx - Q(state, action))
        w += delta * w.grad
        w.grad.zero_()
        if done or truncated:
            s_prime, _ = env.reset()



def sarsa(env: gym.Env, n_steps: int, n_features: int, 
        x: callable = None, gamma = .99, alpha = 0.01, 
        initial_w = None, epsilon: float = 0.5, final_epsilon: float = 0.1):
    """On Policy learning strategy throught SARSA

    Args:
        env (gym.Env): Gymnasium environment with a 
        n_steps (int): Training iterations
        n_features (int): Size of the features vector
        x (callable, optional): A function to describe the state as a vector of features. Defaults to None is the environment already does so
        gamma (float, optional): discount factor. Defaults to .99.
        alpha (float, optional): learning rate. Defaults to 0.01.
        initial_w (_type_, optional): starting weights. Defaults to None.
    """
    epsilon_decay = epsilon / (n_steps / 2)
    
    if x is None:
        x = lambda features: torch.tensor(features, dtype=float, requires_grad=False)
    
    if initial_w is None:
        i_w = np.ones((n_features, env.action_space.n))
    else:
        i_w = initial_w
    
    w = torch.tensor(i_w, dtype=float, requires_grad=True)
    
    def q(state, action):
        return x(state) @ w[:, action]
    
    def policy(state):
        """epsilon-greedy policy"""
        if np.random.random() < epsilon:
            a = env.action_space.sample()
        else:
            a = int(np.argmax(np.array([q(state, action).detach().numpy() for action in range(env.action_space.n)])))
        return a
    
    def update(state, action, reward, s_prime):
        q_target = reward + gamma * q(s_prime, policy(s_prime))
        q_value = q(state, action)
        delta = q_target - q_value
        q_value.backward()
        nonlocal w
        with torch.no_grad():
            w += alpha * delta * w.grad
            w.grad.zero_()
    
    # init
    done = False
    history = []
    rewards = []
    state, _ = env.reset()
    
    # training
    run = range(n_steps)
    for _ in tqdm(run):
        action = policy(state)
        s_prime, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        history.append(w.detach().numpy())
        rewards.append(reward)
        update(state, action, reward, s_prime)
        if done:
            s_prime, _ = env.reset()
            epsilon = max(final_epsilon, epsilon - epsilon_decay)
        state = s_prime
    return w, history, rewards