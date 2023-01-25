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
    policy_history = []
    value_history = []
    for state in mdp.states():
        Vopt[state] = 0

    def Qopt(s, a):
        return sum(p * (r + mdp.gamma * Vopt[s_prime]) for s_prime, p, r in mdp.successors(s, a))

    for iteration in range(MAX_ITERATIONS):
        new_values = {}
        for state in mdp.states():
            if mdp.end(state):
                new_values[state] = 0.
            else:
                new_values[state] = max(Qopt(state, a) for a in mdp.actions(state))
        if max(np.abs(new_values[s] - Vopt[s]) for s in mdp.states()) <= epsilon:
            break
        value_history.append(Vopt)
        Vopt = new_values

        # Collect policy actions
        policy = {}
        for state in mdp.states():
            if mdp.end(state):
                chosen_action = None
            else:
                chosen_action = max((Qopt(state, action), action) for action in mdp.actions(state))[1]
            policy[state] = chosen_action
        policy_history.append(policy)

    return Vopt, policy_history[-1], value_history, policy_history


def policy_iteration(mdp: models.MDP, epsilon: float = 1e-10):
    """
    1) Initialize pi_s with an arbitrary choice of actions from A
    2) Run policy evaluation for the current policy
    3) Update policy: check if the current policy is the value-maximizing action
        If not, update the policy and repeat step 2
    :param mdp: the MDP model
    :param epsilon: termination
    :return: Selected policy
    """
    ## Initialize the policy randomly
    state_actions = {}
    history = []
    for s in mdp.states():
        if len(mdp.actions(s)) > 0:
            a = np.random.choice(list(mdp.actions(s)))
        else:
            a = None
        state_actions[s] = a

    pi = models.StationaryPolicy(state_actions, mdp)

    for iteration in range(MAX_ITERATIONS):
        # Policy evaluation
        optimal_policy = True # we assume this is the optimal policy
        V = policy_evaluation(pi, mdp, epsilon)
        history.append((dict(list(
            state_actions.items())), V)) # keep track of what we do just for the records
        # Now we run policy update
        for s in mdp.states():
            current_value = V[s]
            for a in mdp.actions(s):
                new_value = sum(p * (r + mdp.gamma * V[s_prime]) for
                                s_prime, p, r in mdp.successors(s, a))
                if new_value > current_value and state_actions[s] != a: # We found a new better action
                    state_actions[s] = a
                    current_value = new_value
                    optimal_policy = False
        pi = models.StationaryPolicy(state_actions, mdp)
        if optimal_policy: # nothing has changed, we can stop
            break
    return pi, history


