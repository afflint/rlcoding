# utility functions for MDP
import networkx as nx
import pandas as pd
import pyvis
from IPython.core.display import clear_output, display
from pyvis.network import Network

from models import MDP


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


def show_value_iterations(value_history, policy_history):
    for i, v in enumerate(value_history):
        pi = policy_history[i]
        c = []
        for state, value in v.items():
            a = pi[state]
            c.append({'S': state, 'V': value, 'A': a})
        clear_output(wait=True)
        display(pd.DataFrame(c))
        cmd = input()
        if cmd == 'quit':
            break


def mdp_to_graph(mdp: MDP, state_node_size: int=10, q_node_size: int=5):
    G = nx.DiGraph()
    node2id = dict([(state, i) for i, state in enumerate(mdp.states())])
    t2id = {}
    c = len(mdp.states())
    for state in mdp.states():
        for action in mdp.actions(state):
            t2id[(state, action)] = c
            c += 1
    for state in mdp.states():
        G.add_node(node2id[state], label=state, type='state', color='#999900',
                   size=state_node_size)
        for action in mdp.actions(state):
            G.add_node(t2id[(state, action)], label="({}, {})".format(state, action),
                       type='transition_state', color='#FF9900', size=q_node_size)
            G.add_edge(node2id[state], t2id[(state, action)], type='state_action',
                       action=action, label="{}".format(action))
            for s_prime, p, r in mdp.successors(state, action):
                G.add_node(node2id[s_prime], label=s_prime, type='state',
                           size=state_node_size)
                if p > 0:
                    G.add_edge(t2id[(state, action)], node2id[s_prime],
                               type='transition', p=p, r=r, label="{},{}".format(
                            round(p, 2), round(r, 2)
                        ))
    return G


def plot_mdp(G):
    net = Network("600px", "1200px", notebook=True, directed=True)
    if pyvis._version.__version__ > '0.1.9':
        net.from_nx(G, show_edge_weights=False)
    else:
        net.from_nx(G)
    return net
