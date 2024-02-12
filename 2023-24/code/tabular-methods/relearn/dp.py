"""Dynamic programming algorithms
"""
from relearn.mdp import MDP
import numpy as np

def policy_evaluation(mdp: MDP, gamma: float = 0.99, theta: float = 0.001):
    history = []
    V = np.random.rand(len(mdp.environment.states))
    for state in mdp.environment.states:
        if state.end_state:
            V[state.idx] = 0
    while True:
        Delta = 0
        for state in mdp.environment.states:
            v = V[state.idx].copy()
            V[state.idx] = sum(
                mdp.agent.policy.pmf(action=action, state=state) * 
                sum(
                    mdp.environment.state_reward_proba(state=state, reward=reward, action=action, next_state=s_prime) *
                    (reward.value + gamma * V[s_prime.idx])
                    for s_prime in mdp.environment.states
                    for reward in mdp.environment.rewards
                )
                for action in state.actions
            )
            Delta = max([Delta, np.abs(v - V[state.idx])])
        history.append(Delta)
        if Delta < theta:
            break
    return V, history


def policy_iteration(mdp: MDP, gamma: float = 0.99, theta: float = 0.001):
    history = []
    V = np.random.rand(len(mdp.environment.states))
    for state in mdp.environment.states:
        if state.end_state:
            V[state.idx] = 0
    pi = dict([(s, np.random.choice(mdp.environment.actions)) for s in mdp.environment.states])
    
    # Compute state action value
    def Q(state, action):
        return sum(
            mdp.environment.state_reward_proba(state=state, reward=reward, action=action, next_state=s_prime) *
            (reward.value + gamma * V[s_prime.idx])
            for s_prime in mdp.environment.states
            for reward in mdp.environment.rewards
        )
    
    # Evaluate states under current policy
    def policy_evaluation():
        while True:
            Delta = 0
            for state in mdp.environment.states:
                v = V[state.idx].copy()
                V[state.idx] = Q(state, pi[state])
                Delta = max([Delta, np.abs(v - V[state.idx])])
            if Delta < theta:
                break
    
    # Run policy iteration
    while True:
        history.append(dict([(s, a) for s, a in pi.items()]))
        policy_updated = False 
        for state in mdp.environment.states:
            old_action = pi[state]
            value_actions = np.array([Q(state, a) for a in mdp.environment.actions])
            new_action = mdp.environment.actions[np.argmax(value_actions)]
            pi[state] = new_action # Update policy
            if old_action != new_action:
                policy_updated = True
        if policy_updated:
            policy_evaluation()
        else:
            break
    return pi, V, history

def value_iteration(mdp: MDP, gamma: float = 0.99, theta: float = 0.001):
    history = []
    V = np.random.rand(len(mdp.environment.states))
    for state in mdp.environment.states:
        if state.end_state:
            V[state.idx] = 0
    
    # Compute state action value
    def Q(state, action):
        return sum(
            mdp.environment.state_reward_proba(state=state, reward=reward, action=action, next_state=s_prime) *
            (reward.value + gamma * V[s_prime.idx])
            for s_prime in mdp.environment.states
            for reward in mdp.environment.rewards
        )
    
    # Evaluate states
    while True:
        Delta = 0
        for state in mdp.environment.states:
            v = V[state.idx].copy()
            V[state.idx] = max([Q(state, a) for a in mdp.environment.actions])
            Delta = max([Delta, np.abs(v - V[state.idx])])
        if Delta < theta:
            break
    
    # Get the policy
    pi = {}
    for state in mdp.environment.states:
        actions_value = np.array([Q(state, a) for a in mdp.environment.actions])
        pi[state] = mdp.environment.actions[np.argmax(actions_value)]
    return pi, V, history