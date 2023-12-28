import numpy as np


ATOL = np.sqrt(np.finfo(np.float64).eps)


def kahansum(input):
    summ = c = 0
    for num in input:
        y = num - c
        t = summ + y
        c = (t - summ) - y
        summ = t
    return summ