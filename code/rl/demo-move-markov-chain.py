from collections import defaultdict
import numpy as np
import pandas as pd
from rl.scratch.markov import  MarkovChain


def main(areas: int):
    """
    We have an object moving on a line left or right
    States are areas along the line
    Transitions are areas you end to
    :return: None
    """
    mc = MarkovChain(number_of_states=areas)
    mc.set_transition(0, np.array([.6, .3, .1]))
    mc.set_transition(1, np.array([.6, .3, .1]))
    mc.set_transition(2, np.array([.1, .1, .8]))
    return mc


if __name__ == "__main__":
    mc = main(3)
    iterations = 1000
    for i in range(iterations):
        mc.step()
    stats = defaultdict(lambda: 0)
    for h in mc.history:
        stats[h] += 1
    S = pd.Series(stats).sort_values(ascending=False)
    print(S / iterations)