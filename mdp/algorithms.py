# Algorithms for Policy Evaluation and Value Iteration
import numpy as np

import models

MAX_ITERATIONS = 1000


def policy_evaluation(policy: models.Policy,
                      mdp: models.MDP, epsilon: float = 1e-10):
    """
    V_t(s) = Q_{t-1}(s, pi_s)
    Q_{t-1}(s, pi_s) = sum_s' T(s, a, s')[reward(s, pi_s, s') + gamma * V_{t-1}(s')
    :param policy: the policy under evaluation
    :param mdp: model
    :param epsilon: threshold
    :return: the expected utility for each state
    """
    V = dict([(state, 0) for state in mdp.states()])

    def Q(s, a):
        return sum(p * (r + mdp.gamma * V[s_prime]) for s_prime, p, r in mdp.successors(s, a))

    for iteration in range(MAX_ITERATIONS):
        new_values = {}
        for state in mdp.states():
            if mdp.end(state):
                new_values[state] = 0.
            else:
                new_values[state] = Q(state, policy.action(state))
        if max(np.abs(new_values[s] - V[s]) for s in mdp.states()) <= epsilon:
            break
        V = new_values

    return V


def value_iteration(mdp: models.MDP, epsilon: float = 1e-10):
    """
    Vopt_t(s) = max_a Qopt_{t-1}(s, a)
    Qopt_{t-1}(s, a) = sum_s' T(s, a, s')[reward(s, a, s') + gamma * Vopt_{t-1}(s')
    :param mdp: model
    :param epsilon: threshold
    :return: the expected utility for each state
    """
    Vopt = {}
    for state in mdp.states():
        Vopt[state] = (0, None)

    def Q(s, a):
        return sum(p * (r + mdp.gamma * Vopt[s_prime][0]) for s_prime, p, r in mdp.successors(s, a))

    for iteration in range(MAX_ITERATIONS):
        new_values = {}
        for state in mdp.states():
            if mdp.end(state):
                new_values[state] = (0., None)
            else:
                new_values[state] = max((Q(state, a), a) for a in mdp.actions(state))
        if max(np.abs(new_values[s][0] - Vopt[s][0]) for s in mdp.states()) <= epsilon:
            break
        Vopt = new_values

    return Vopt
