"""Implementing main algorithms with GyM for the GridSuttonBarto example
"""
from mdp.environments import GridSuttonBarto
import numpy as np


def policy_evaluation(env: GridSuttonBarto, gamma: float = 0.99, theta: float = 0.001):
    history = []
    V = np.random.rand(env.observation_space.n)
    for state in range(env.observation_space.n):
        if state == env.unwrapped.size - 1 or state == 0:
            V[state] = 0
    while True:
        Delta = 0
        for state in range(env.observation_space.n):
            v = V[state].copy()
            V[state] = sum(
                (1 / env.action_space.n) * 
                sum(
                    env.unwrapped.proba(state=state, action=action, s_prime=s_prime) *
                    (env.unwrapped.reward(s_prime) + gamma * V[s_prime])
                    for s_prime in range(env.observation_space.n)
                )
                for action in range(env.action_space.n)
            )
            Delta = max([Delta, np.abs(v - V[state])])
        history.append(Delta)
        if Delta < theta:
            break
    return V, history


def value_iteration(env: GridSuttonBarto, gamma: float = 0.99, theta: float = 0.001):
    history = []
    V = np.random.rand(env.observation_space.n)
    for state in range(env.observation_space.n):
        if state == env.unwrapped.size - 1 or state == 0:
            V[state] = 0
    
    # Compute state action value
    def Q(state, action):
        return sum(
            env.unwrapped.proba(state=state, action=action, s_prime=s_prime) *
            (env.unwrapped.reward(s_prime) + gamma * V[s_prime])
            for s_prime in range(env.observation_space.n)
        )
    
    # Evaluate states
    while True:
        Delta = 0
        for state in range(env.observation_space.n):
            v = V[state].copy()
            V[state] = max([Q(state, a) for a in range(env.action_space.n)])
            Delta = max([Delta, np.abs(v - V[state])])
        if Delta < theta:
            break
    
    # Get the policy
    pi = {}
    for state in range(env.observation_space.n):
        actions_value = np.array([Q(state, a) for a in range(env.action_space.n)])
        pi[state] = np.argmax(actions_value)
    return pi, V, history