import sys
from collections import defaultdict
import numpy as np
import copy


def Qlearning(mdp,
              num_episodes: int = 1000,
              max_steps: int = 100,
              min_diff: float = 1e-3,
              eta: float = 0.01):
    # Init the Q(S,A) table
    Q = defaultdict(lambda: defaultdict(lambda: float('-inf')))
    for state in mdp.states():
        for action in mdp.actions(state):
            Q[state][action] = 0
    # set up a history
    history = []
    total_reward_per_episode = []
    for episode in range(num_episodes):
        R = 0
        Q_old = copy.deepcopy(Q)
        state = np.random.choice(list(mdp.start()))
        for step in range(max_steps):
            action = np.random.choice(list(mdp.actions(state))) # random policy (try greedy)
            # Take action
            successors = [x for x in mdp.successors(state, action)]
            if len(successors) > 0:
                p = [p for _, p, _ in successors]
                s_prime, _, reward = successors[np.random.choice(range(len(successors)), p=p)]
                R += reward
                # Update
                Q[state][action] = (1-eta) * Q_old[state][action] + eta * (
                    reward + mdp.gamma * max(Q_old[s_prime].values())
                )
                state = s_prime
                if mdp.end(state):
                    break
            else:
                break
        history.append(copy.deepcopy(Q_old))
        total_reward_per_episode.append(R)
#        diff = 0
#        for k, v in Q.items():
#            for x, y in v.items():
#                diff += np.abs(y - Q_old[k][x])
#        if diff < min_diff:
#            break
    return Q, history, total_reward_per_episode


def sarsa(mdp,
              num_episodes: int = 1000,
              max_steps: int = 100,
              min_diff: float = 1e-3,
              eta: float = 0.01):
    # Init the Q(S,A) table
    Q = defaultdict(lambda: defaultdict(lambda: float('-inf')))
    for state in mdp.states():
        for action in mdp.actions(state):
            Q[state][action] = 0

    # Counter for visits to the states
    N = dict([(s, 1) for s in mdp.states()])

    # Set up functions for choosing the policy
    def epsilon(state):
        return 1 / N[state]

    def epsilon_greedy(state):
        x = np.random.uniform()
        if x < epsilon(state):
            action = np.random.choice(list(mdp.actions(state)))
        else:
            action = max([(v, a) for a, v in Q[state].items()])[1]
        return action

    # set up a history
    history = []
    total_reward_per_episode = []
    for episode in range(num_episodes):
        R = 0
        Q_old = copy.deepcopy(Q)
        state = np.random.choice(list(mdp.start()))
        action = epsilon_greedy(state)
        for step in range(max_steps):
            # Take action
            successors = [x for x in mdp.successors(state, action)]
            if len(successors) > 0:
                p = [p for _, p, _ in successors]
                s_prime, _, reward = successors[np.random.choice(range(len(successors)), p=p)]
                R += reward

                # Now observe what happens if you use your policy on s_prime
                a_prime = epsilon_greedy(s_prime)

                # Update
                Q[state][action] = (1-eta) * Q_old[state][action] + eta * (
                    reward + mdp.gamma * Q_old[s_prime][a_prime]
                )
                state = s_prime
                action = a_prime
                if mdp.end(state):
                    break
            else:
                break
        history.append(copy.deepcopy(Q_old))
        total_reward_per_episode.append(R)
#        diff = 0
#        for k, v in Q.items():
#            for x, y in v.items():
#                diff += np.abs(y - Q_old[k][x])
#        if diff < min_diff:
#            break
    return Q, history, total_reward_per_episode
