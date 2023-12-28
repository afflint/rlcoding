from rl.mdp.model import MDP
from rl.mdp.policy import Policy
from collections import defaultdict
import numpy as np


def montecarlo_value_function(policy: Policy, model: MDP, first_visit: bool = True, max_iterations: int = 1000) -> np.ndarray:
    """
    MonteCarlo Value Function On-Policy estimation

    Args:
        policy (Policy): Policy
        model (MDP): The MDP
        first_visit (bool, optional): Use First-Visit MC. Defaults to True.
        max_iterations (int, optional): Max interations. Defaults to 1000.

    Returns:
        np.ndarray: Value Function
    """
    states = sorted(list(model.states))
    V = np.zeros(len(states))
    visits = defaultdict(list)
    for i in range(max_iterations):
        visited = set()
        episode = policy.episode()
        G = 0
        for s, a, r in reversed(episode):
            G = model.gamma * G + r
            if first_visit and s in visited:
                pass
            else:
                visits[s].append(G)
                visited.add(s)
    for state in states:
        if len(visits[state]) > 0:
            V[states.index(state)] = np.array(visits[state]).mean()
        else:
            V[states.index(state)] = np.nan
    return V