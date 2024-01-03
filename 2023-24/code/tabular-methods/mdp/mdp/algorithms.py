from typing import Tuple
import numpy as np
from mdp.policy import Policy, StationaryPolicy
from mdp.model import MDP


MAX_ITERATIONS = 1000


def policy_evaluation(policy: Policy, mdp: MDP, epsilon: float = 1e-10) -> np.ndarray:
    """
    V_t(s) = Q_{t-1}(s, pi_s)
    Q_{t-1}(s, pi_s) = sum_s' T(s, a, s')[reward(s, pi_s, s') + gamma * V_{t-1}(s')

    Args:
        policy (Policy): The policy used for selecting actions
        mdp (MDP): The MDP environment
        epsilon (float, optional): Condition to break. Defaults to 1e-10.

    Returns:
        np.ndarray: V(s) for all S
    """
    V = np.zeros(len(mdp.states))
    
    def Q(s, a):
        return sum(p * (r + mdp.gamma * V[s_prime]) for s_prime, p, r in mdp.successors(s, a))

    for _ in range(MAX_ITERATIONS):
        new_V = np.zeros(len(V))
        for state in range(mdp.T.shape[0]):
            if mdp.is_terminal(state):
                new_V[state] = 0.
            else:
                new_V[state] = Q(state, policy[state])
        if max(np.abs(new_V[s] - V[s]) for s in range(mdp.T.shape[0])) <= epsilon:
            break
        V = np.array([x for x in new_V])
    return V


def value_iteration(mdp: MDP, epsilon: float = 1e-10) -> Tuple[np.array, dict, list, list]:
    """
    V_opt_t(s) = max_a Q_opt_{t-1}(s, a)
    Q_opt_{t-1}(s, a) = sum_s' T(s, a, s')[reward(s, a, s') + gamma * V_opt_{t-1}(s')

    Args:
        mdp (MDP): The MDP environment
        epsilon (float, optional): Condition to break. Defaults to 1e-10.

    Returns:
        Tuple[np.array, dict, list, list]: Optimal values, optimal policy, value history, policy history
    """
    V_opt = np.zeros(len(mdp.states))
    policy_history = []
    value_history = []

    def Q_opt(s, a):
        return sum(p * (r + mdp.gamma * V_opt[s_prime]) for s_prime, p, r in mdp.successors(s, a))

    for _ in range(MAX_ITERATIONS):
        new_values = np.zeros(len(V_opt))
        for state in range(mdp.T.shape[0]):
            if mdp.is_terminal(state):
                new_values[state] = 0.
            else:
                new_values[state] = max(Q_opt(state, a) for a in range(mdp.T.shape[1]))
        if max(np.abs(new_values[s] - V_opt[s]) for s in range(mdp.T.shape[0])) <= epsilon:
            break
        value_history.append(V_opt)
        V_opt = np.array([x for x in new_values])

        # Collect policy actions
        policy = {}
        for state in range(mdp.T.shape[0]):
            if mdp.is_terminal(state):
                chosen_action = None
            else:
                chosen_action = max((Q_opt(state, action), action) for action in range(mdp.T.shape[1]))[1]
            policy[state] = chosen_action
        policy_history.append(policy)

    return V_opt, policy_history[-1], value_history, policy_history


def policy_iteration(mdp: MDP, epsilon: float = 1e-10) -> StationaryPolicy:
    """
    1) Initialize pi_s with an arbitrary choice of actions from A
    2) Run policy evaluation for the current policy
    3) Update policy: check if the current policy is the value-maximizing action
        If not, update the policy and repeat step 2

    Args:
        mdp (MDP): MDP model
        epsilon (float, optional): Stop iteration factor. Defaults to 1e-10.

    Returns:
        StationaryPolicy: Map state action
    """
    ## Initialize the policy randomly
    state_actions = {}
    history = []
    for s in mdp.states:
        actions = [a for a in mdp.actions if len(list(mdp.successors(s, a))) > 0]
        if len(actions) > 0:
            state_actions[s] = np.random.choice(actions)
        else:
            state_actions[s] = None
    pi = StationaryPolicy(mdp, state_action=state_actions)
    
    def check_update(new_value, old_value, s, a):
        if new_value > old_value and np.abs(new_value - old_value) > epsilon and state_actions[s] != a:
            return True
        else:
            return False
        
    def check_break():
        if len(history) > 1:
            if history[-1][0] == history[-2][0]:
                if max(np.abs(history[-1][1][s] - history[-2][1][s]) for s in range(mdp.T.shape[0])) <= epsilon:
                    return True
            else:
                return False
        else:
            return False

    for iteration in range(MAX_ITERATIONS):
        # Policy evaluation
        optimal_policy = True # we assume this is the optimal policy
        V = policy_evaluation(policy=pi, mdp=mdp, epsilon=epsilon)
        history.append((dict(list(
            state_actions.items())), V)) # keep track of what we do just for the records
        # Now we run policy update
        for s in mdp.states:
            current_value = V[s]
            for a in mdp.actions:
                successors = list(mdp.successors(s, a))
                if len(successors) > 0:
                    new_value = sum(p * (r + mdp.gamma * V[s_prime]) for s_prime, p, r in successors)
                    if check_update(new_value, current_value, s, a): # We found a new better action
                        state_actions[s] = a
                        current_value = new_value
                        optimal_policy = False
        pi = StationaryPolicy(mdp, state_actions)
        if optimal_policy or check_break(): # nothing has changed, we can stop
            break
    return pi, history