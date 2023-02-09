import pandas as pd
from IPython.core.display import clear_output, display


def show_q_learning(history):
    for i, Q in enumerate(history):
        c = pd.DataFrame(Q).T
        clear_output(wait=True)
        display(c)
        cmd = input()
        if cmd == 'quit':
            break
