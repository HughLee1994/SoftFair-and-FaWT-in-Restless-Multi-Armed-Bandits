import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import math
import matplotlib.pyplot as plt
import glob
import time

from utils import precomputeBelief, getThresholdC


def whittleIndex(T, L=180, ba=[], bna=[], limit_a=None, limit_na=None):
    '''
    Pre-compute whittle's index for all possible values of x1 and all values of x2.
    If limit_a is not None, then compute only until limit_a for A chain.
    If limit_na is not NOne, then compute only until limit_na for NA chain.

    T should be such that T[0]= T_passive and T[1] is T_active
    w1 stores whittle index for current belief state for the A chain
    w2 stores whittle index for current belief state for the NA chain
    '''

    if len(ba) == 0 or len(bna) == 0:
        ba, bna = precomputeBelief(T[0][1][1], T[0][0][1], T[1][1][1], T[1][0][1], L=L)

    w1 = np.zeros(L)
    w2 = np.zeros(L)

    x1_current = 1
    x2_current = 1

    if limit_a is None:
        limit_a = L - 1

    if limit_na is None:
        limit_na = L - 1

    while (x1_current <= limit_a or x2_current <= limit_na):

        # print (x1_current, x2_current)

        if not x1_current < L:  # x1 is full only look for x2(down) whittle index
            c_down = getThresholdC((x1_current, x2_current), (x1_current, x2_current + 1), T[0], T[1], ba=ba, bna=bna)
            w2[x2_current - 1] = c_down
            x2_current += 1

        elif not x2_current < L:  # x2 is full only look for x1(right) whittle index
            c_right = getThresholdC((x1_current, x2_current), (x1_current + 1, x2_current), T[0], T[1], ba=ba, bna=bna)
            w1[x1_current - 1] = c_right
            x1_current += 1

        else:  # neither directions is full, look for both whittle index
            c_down = getThresholdC((x1_current, x2_current), (x1_current, x2_current + 1), T[0], T[1], ba=ba, bna=bna)
            c_right = getThresholdC((x1_current, x2_current), (x1_current + 1, x2_current), T[0], T[1], ba=ba, bna=bna)

            if c_down < c_right:
                w2[x2_current - 1] = c_down
                x2_current += 1
            else:
                w1[x1_current - 1] = c_right
                x1_current += 1

    return w1, w2






