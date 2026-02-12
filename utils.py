import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import glob
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib import rcParams

import matplotlib as mpl


def returnKGreatestIndices_(arr, k):
    arr = np.array(arr)
    ans = arr.argsort()[-k:][::-1]
    return ans


def returnKGreatestIndices(fairness_constraints, arr, k):
    arr = np.array(arr)
    # print('shape', np.shape(arr))
    # ans = arr.argsort()[-k:][::-1]

    ans = np.ones(k)
    n = 0
    for i in range(len(fairness_constraints)):
        if fairness_constraints[i] <= 0 and n<k:
            ans[n] = i
            n += 1
    if n >= k:
        n = k
    k_remain = k - n
    if k_remain>0:
        ans[n:] = arr.argsort()[-k_remain:][::-1]
    ans = ans.astype(int)
    return ans


def generateRandomTmatrix(N, random_stream):

    # Return a randomly generated T matrix (not unformly random because of sorting)
    T=np.zeros((N,2,2,2))
    for i in range(N):
        p_pass_01, p_pass_11, p_act_01, p_act_11=sorted(random_stream.uniform(size=4))
        T[i,0]=np.array([[1-p_pass_01, p_pass_01],[1-p_pass_11, p_pass_11]])
        T[i,1]=np.array([[1-p_act_01, p_act_01],[1-p_act_11, p_act_11]])
    return T


def precomputeBelief(p11, p01, p11active, p01active, L=180):
    # precomputeBelief(T[0][1][1],T[0][0][1],T[1][1][1],T[1][0][1])
    bA = np.zeros(L)
    bNA = np.zeros(L)

    bA[-1] = 1.
    bNA[-1] = 0.

    for t in range(L):
        if t == 0:
            bA[t] = bA[t - 1] * p11active + (1. - bA[t - 1]) * p01active
            bNA[t] = bNA[t - 1] * p11active + (1. - bNA[t - 1]) * p01active
        else:
            bA[t] = bA[t - 1] * p11 + (1. - bA[t - 1]) * p01
            bNA[t] = bNA[t - 1] * p11 + (1. - bNA[t - 1]) * p01
    return bA, bNA


def getThresholdC(point1, point2, Tpass, Tact, ba=[], bna=[]):
    if len(ba) == 0 or len(bna) == 0:
        ba, bna = precomputeBelief(Tpass[1][1], Tpass[0][1], Tact[1][1], Tact[0][1])

    # print (point1, point2)
    slope1, const1 = Cavg(point1[0], point1[1], Tpass=Tpass, Tact=Tact, ba=ba, bna=bna)
    slope2, const2 = Cavg(point2[0], point2[1], Tpass=Tpass, Tact=Tact, ba=ba, bna=bna)
    c_threshold = (const1 - const2) / (slope2 - slope1)

    return c_threshold


def Cavg(x1, x2, c0=None, Tpass=np.identity(2), Tact=np.identity(2), ba=[], bna=[]):
    if len(ba) == 0 or len(bna) == 0:
        ba, bna = precomputeBelief(Tpass[1][1], Tpass[0][1], Tact[1][1], Tact[0][1])

    q = ((x1 * bna[x2 - 1]) / (1 - ba[x1 - 1]) + x2) ** -1
    p = q * (bna[x2 - 1] / (1 - ba[x1 - 1]))

    if c0:
        cavg = p * (x1 - np.sum(ba[:x1])) + q * (x2 - np.sum(bna[:x2])) + (p + q) * c0
        return cavg, p * (x1 - np.sum(ba[:x1])) + q * (x2 - np.sum(bna[:x2])), (p + q)
        # return cavg, (p+q), p*(x1- np.sum(ba[:x1]))+q*(x2- np.sum(bna[:x2]))
    else:
        # return slope, intercept
        return (p + q), p * (x1 - np.sum(ba[:x1])) + q * (x2 - np.sum(bna[:x2]))


def verify_T_matrix(T):

    valid = True
    # print(T[0, 0, 1], T[0, 1, 1])
    valid &= T[0, 0, 1] <= T[0, 1, 1] # non-oscillate condition
    # print(valid)
    valid &= T[1, 0, 1] <= T[1, 1, 1] # must be true for active as well
    # print(valid)
    valid &= T[0, 1, 1] <= T[1, 1, 1] # action has positive "maintenance" value
    # print(valid)
    valid &= T[0, 0, 1] <= T[1, 0, 1] # action has non-negative "influence" value
    # print(valid)
    return valid


def epsilon_clip(T, epsilon):
    return np.clip(T, epsilon, 1-epsilon)


def barPlot(labels, values, errors, ylabel='Average Adherence out of 180 days',
            title='Adherence simulation for 20 patients/4 calls', filename='image.png', root='.',
            bottom=0):
    fname = os.path.join(root, filename)
    # plt.figure(figsize=(8,6))
    x = np.arange(len(labels))  # the label locations
    width = 0.85  # the width of the bars
    fig, ax = plt.subplots(figsize=(8, 5))
    # rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
    rects1 = ax.bar(x, values, width, bottom=bottom, label='Intervention benefit')

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


def soft_max(x):

    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def soft_max_stable(x):

    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def prob_cal(x, k):
    prob = np.zeros(len(x))
    for i in range(len(x)):
        prob[i] = 1 - math.pow((1-x[i]), k)
    return prob

