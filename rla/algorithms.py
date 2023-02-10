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
        R = defaultdict(lambda : 0)
        state = np.random.choice(list(mdp.start()))
        for step in range(max_steps):
            history.append(copy.deepcopy(Q))
            action = np.random.choice(list(mdp.actions(state))) # random policy (try greedy)
            # Take action
            successors = [x for x in mdp.successors(state, action)]
            if len(successors) > 0:
                p = [p for _, p, _ in successors]
                s_prime, _, reward = successors[np.random.choice(range(len(successors)), p=p)]
                R[action] += reward
                # Update
                Q[state][action] = (1-eta) * Q[state][action] + eta * (
                    reward + mdp.gamma * max(Q[s_prime].values())
                )
                state = s_prime
                if mdp.end(state):
                    break
            else:
                break
        total_reward_per_episode.append(copy.deepcopy(R))
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
    def epsilon(s_check):
        return 1 / N[s_check]

    def epsilon_greedy(s_check):
        x = np.random.uniform()
        if x < epsilon(s_check):
            to_action = np.random.choice(list(mdp.actions(s_check)))
        else:
            to_action = max([(v, a) for a, v in Q[s_check].items()])[1]
        return to_action

    # set up a history
    history = []
    total_reward_per_episode = []
    for episode in range(num_episodes):
        R = defaultdict(lambda : 0)
        state = np.random.choice(list(mdp.start()))
        action = epsilon_greedy(state)
        for step in range(max_steps):

            history.append(copy.deepcopy(Q))

            # Take action
            successors = [x for x in mdp.successors(state, action)]
            if len(successors) > 0:
                p = [p for _, p, _ in successors]
                s_prime, _, reward = successors[np.random.choice(range(len(successors)), p=p)]
                R[action] += reward

                # Now observe what happens if you use your policy on s_prime
                a_prime = epsilon_greedy(s_prime)

                # Update
                Q[state][action] = (1-eta) * Q[state][action] + eta * (
                    reward + mdp.gamma * Q[s_prime][a_prime]
                )
                state = s_prime
                action = a_prime
                if mdp.end(state):
                    break
            else:
                break
        total_reward_per_episode.append(copy.deepcopy(R))
#        diff = 0
#        for k, v in Q.items():
#            for x, y in v.items():
#                diff += np.abs(y - Q_old[k][x])
#        if diff < min_diff:
#            break
    return Q, history, total_reward_per_episode


def sarsa_lambda(mdp,
                 lambda_: float = .8,
                 num_episodes: int = 1000,
                 max_steps: int = 100,
                 min_diff: float = 1e-3,
                 eta: float = 0.01):
    # Init the Q(S,A) table
    Q = defaultdict(lambda: defaultdict(lambda: float('-inf')))
    for state in mdp.states():
        for action in mdp.actions(state):
            Q[state][action] = 0

    # Eligibility trace
    E = defaultdict(lambda: defaultdict(lambda: 0))
    for state in mdp.states():
        for action in mdp.actions(state):
            E[state][action] = 0

    # Counter for visits to the states
    N = dict([(s, 1) for s in mdp.states()])

    # Set up functions for choosing the policy
    def epsilon(s_check):
        return 1 / N[s_check]

    def epsilon_greedy(s_check):
        x = np.random.uniform()
        if x < epsilon(s_check):
            to_action = np.random.choice(list(mdp.actions(s_check)))
        else:
            to_action = max([(v, a) for a, v in Q[s_check].items()])[1]
        return to_action

    # set up a history
    history = []
    total_reward_per_episode = []
    for episode in range(num_episodes):
        R = defaultdict(lambda : 0)
        state = np.random.choice(list(mdp.start()))
        action = epsilon_greedy(state)

        for step in range(max_steps):

            history.append(copy.deepcopy(Q))

            # Take action
            successors = [x for x in mdp.successors(state, action)]
            if len(successors) > 0:
                p = [p for _, p, _ in successors]
                s_prime, _, reward = successors[np.random.choice(range(len(successors)), p=p)]
                R[action] += reward

                # Now observe what happens if you use your policy on s_prime
                a_prime = epsilon_greedy(s_prime)

                # Compute delta
                delta = reward + mdp.gamma * Q[s_prime][a_prime] - Q[state][action]
                E[state][action] = E[state][action] + 1

                # Update
                for state in mdp.states():
                    for action in mdp.actions(state):
                        Q[state][action] = Q[state][action] + eta * delta * E[state][action]
                        E[state][action] = mdp.gamma * lambda_ * E[state][action]

                state = s_prime
                action = a_prime
                if mdp.end(state):
                    break
            else:
                break
        total_reward_per_episode.append(copy.deepcopy(R))
#        diff = 0
#        for k, v in Q.items():
#            for x, y in v.items():
#                diff += np.abs(y - Q_old[k][x])
#        if diff < min_diff:
#            break
    return Q, history, total_reward_per_episode

