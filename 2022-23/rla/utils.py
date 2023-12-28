from collections import defaultdict

import numpy as np
import pandas as pd
from IPython.core.display import clear_output, display
from matplotlib import pyplot as plt


def show_q_learning(history):
    for i, Q in enumerate(history):
        c = pd.DataFrame(Q).T
        clear_output(wait=True)
        display(c)
        cmd = input()
        if cmd == 'quit':
            break


def history_to_tensor(history):
    states = list(history[0].keys())
    actions = list(history[0][states[0]].keys())
    t = np.zeros((len(history), len(states), len(actions)))
    for i, h in enumerate(history):
        for s, state in enumerate(states):
            for a, action in enumerate(actions):
                t[i,s,a] = h[state][action]
    return t, states, actions


def plot_stats(history, rewards):
    tensor, states, actions = history_to_tensor(history)
    q_actions = {}
    for i, s in enumerate(states):
        for j, a in enumerate(actions):
            q_actions["{}{}".format(s, a)] = tensor[:, i, j]
    r_actions = defaultdict(list)
    for r in rewards:
        for k, v in r.items():
            r_actions[k].append(v)
    return q_actions, r_actions


def plot_algorithms(history, rewards):
    q_actions, r_actions = plot_stats(history, rewards)
    fig, ax = plt.subplots(figsize=(10, 4), ncols=2)
    for k, v in r_actions.items():
        ax[0].plot(v, label=k)
    ax[0].legend()
    for k, v in q_actions.items():
        ax[1].plot(v, label=k)
    ax[1].legend()
    plt.tight_layout()
    plt.show()