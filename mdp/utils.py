# utility functions for MDP
import pandas as pd


def transitions_table(mdp):
    output = []
    for state in mdp.states():
        for action in mdp.actions(state):
            for state_prime in mdp.states():
                p, r = mdp.transition_reward(state, action, state_prime)
                output.append({
                    'from_state': state, 'action': action, 'to_state': state_prime,
                    'reward': r, 'probability': p
                })
    return pd.DataFrame(output)