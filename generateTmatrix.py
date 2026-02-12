import numpy as np
import pandas as pd
import time
# import pomdp

from itertools import combinations
from whittle import *
from utils import *


import os
import argparse
import tqdm


def generateTmatrixReal(N, file_root='.', responsive_patient_fraction=0.4, epsilon=0.005,
                        shift1=0,shift2=0,shift3=0,shift4=0, intervention_effect=0.05,
                        thresh_opt_frac=None, beta=0.5, quick_check=False):
    """
        Generates a Nx2x2x2 T matrix indexed as: T[patient_number][action][current_state][next_state]
        action=0 denotes passive action, a=1 is active action
        State 0 denotes NA and state 1 denotes A
        """
    fname = os.path.join(file_root + '/data/', 'patient_T_matrices.npy')
    real = np.load(fname)

    T = np.zeros((N, 2, 2, 2))
    # Passive action transition probabilities
    penalty_pass_00 = 0
    penalty_pass_11 = 0

    # Active action transition probabilities
    benefit_act_00 = 0
    benefit_act_11 = 0

    if thresh_opt_frac is None:
        choices = np.random.choice(np.arange(real.shape[0]), N, replace=True)
    else:
        thres_opt_patients = np.random.choice([i for i in range(N)], size=int(thresh_opt_frac * N), replace=False)

    i = 0
    while i < N:

        if thresh_opt_frac is None:
            choice = choices[i]
        else:
            choice = np.random.choice(np.arange(real.shape[0]), 1, replace=True)[0]
        T_base = np.zeros((2, 2))
        T_base[0, 0] = real[choice][0]
        T_base[1, 1] = real[choice][1]
        T_base[0, 1] = 1 - T_base[0, 0]
        T_base[1, 0] = 1 - T_base[1, 1]

        T_base = smooth_real_probs(T_base, epsilon)

        shift = intervention_effect

        # Patient responds well to call
        benefit_act_00 = np.random.uniform(low=0., high=shift)  # will subtract from prob of staying 0,0
        benefit_act_11 = benefit_act_00 + np.random.uniform(low=0., high=shift)  # will add to prob of staying 1,1
        # add benefit_act_00 to benefit_act_11 to guarantee the p11>p01 condition

        # Patient does well on their own, low penalty for not calling
        penalty_pass_11 = np.random.uniform(low=0., high=shift)  # will sub from prob of staying 1,1
        penalty_pass_00 = penalty_pass_11 + np.random.uniform(low=0., high=shift)  # will add to prob of staying 0,0

        '''
        For perturbation experiment only. TEMPORARY CODE below. 
        '''
        """
        benefit_act_00=np.random.uniform(low=0., high=shift1) # will subtract from prob of staying 0,0
        benefit_act_11= benefit_act_00 + np.random.uniform(low=0., high=shift2) # will add to prob of staying 1,1
        # add benefit_act_00 to benefit_act_11 to guarantee the p11>p01 condition


        # Patient does well on their own, low penalty for not calling
        penalty_pass_11=np.random.uniform(low=0., high=shift3) # will sub from prob of staying 1,1
        penalty_pass_00=penalty_pass_11+np.random.uniform(low=0., high=shift4) # will add to prob of staying 0,0
        """

        T_pass = np.copy(T_base)
        T_act = np.copy(T_base)

        T_act[0, 0] = max(0, T_act[0, 0] - benefit_act_00)
        T_act[1, 1] = min(1, T_act[1, 1] + benefit_act_11)

        T_pass[0, 0] = min(1, T_pass[0, 0] + penalty_pass_00)
        T_pass[1, 1] = max(0, T_pass[1, 1] - penalty_pass_11)

        T_pass[0, 1] = 1 - T_pass[0, 0]
        T_pass[1, 0] = 1 - T_pass[1, 1]

        T_act[0, 1] = 1 - T_act[0, 0]
        T_act[1, 0] = 1 - T_act[1, 1]

        T_pass = epsilon_clip(T_pass, epsilon)
        T_act = epsilon_clip(T_act, epsilon)

        # print(T_pass)
        # print(T_act)
        # print()

        if not verify_T_matrix(np.array([T_pass, T_act])):
            print("T matrix invalid\n", np.array([T_pass, T_act]))
            raise ValueError()

        if thresh_opt_frac is None:
            satisfies_condition = True

        else:
            satisfies_condition = False
            if i in thres_opt_patients:  # Threshold opt patient
                satisfies_condition = isThresholdOptimal([T_pass, T_act], beta, quick_check=quick_check)
            else:  # Reverse Threshold opt patient
                satisfies_condition = isReverseThresholdOptimal([T_pass, T_act], beta, quick_check=quick_check)
        if satisfies_condition:
            T[i, 0] = T_pass
            T[i, 1] = T_act
            i += 1
    return T


def generateTmatrixBadf(N, responsive_patient_fraction=0.4,
                        range_pass_00=(0.6, 0.8), range_pass_11=(0.6, 0.89),
                        range_act_g_00=(0, 0.2), range_act_g_11=(0.9, 1.0),
                        range_act_b_00=(0.7, 0.9), range_act_b_11=(0.9, 1.0)):
    # print("p_act01 < p01/(p01+p10)")

    """
    Generates a Nx2x2x2 T matrix indexed as: T[patient_number][action][current_state][next_state]
    action=0 denotes passive action, a=1 is active action
    State 0 denotes NA and state 1 denotes A
    """

    T = np.zeros((N, 2, 2, 2))
    # Passive action transition probabilities
    p_pass_00 = np.random.uniform(low=range_pass_00[0], high=range_pass_00[1], size=N)
    p_pass_11 = np.random.uniform(low=range_pass_11[0], high=range_pass_11[1], size=N)

    # Active action transition probabilities
    # responsive_patient_fraction=0.4
    p_act_00 = np.zeros(N)
    p_act_11 = np.zeros(N)
    for i in range(N):
        if np.random.binomial(1, responsive_patient_fraction) == 1:
            # Patient responds well to call
            p_act_00[i] = np.random.uniform(low=range_act_g_00[0], high=range_act_g_00[1])
            p_act_11[i] = np.random.uniform(low=range_act_g_11[0], high=range_act_g_11[1])

            p_act01 = 1 - p_act_00[i]
            p01 = 1 - p_pass_00[i]
            p10 = 1 - p_pass_11[i]
            if p_act01 < p01 / (p01 + p10):
                raise ValueError("Intended good patient was bad.")
        else:
            # Patient doesn't respond well to call
            p_act_00[i] = np.random.uniform(low=range_act_b_00[0], high=range_act_b_00[1])
            p_act_11[i] = np.random.uniform(low=range_act_b_11[0], high=range_act_b_11[1])

            p_act01 = 1 - p_act_00[i]
            p01 = 1 - p_pass_00[i]
            p10 = 1 - p_pass_11[i]
            if not (p_act01 < p01 / (p01 + p10)):
                raise ValueError("Intended bad patient was good.")

    for i in range(N):
        T[i, 0] = np.array([[p_pass_00[i], 1 - p_pass_00[i]], [1 - p_pass_11[i], p_pass_11[i]]])
        T[i, 1] = np.array([[p_act_00[i], 1 - p_act_00[i]], [1 - p_act_11[i], p_act_11[i]]])

    # print (T[:20])
    return T

def generateDesignedTmatrix(N):
    # Return a randomly generated T matrix (not unformly random because of sorting)
    T = np.zeros((N, 2, 2, 2))


    # for i in range(N):
    #     if i%2==0:
    #         T[i,:,:,:] =  [[[0.8, 0.3],
    #                  [0.3,  0.7]],
    #
    #                 [[0.05, 0.95],
    #                  [0.5, 0.5]]]
    #     else:
    #         T[i,:,:,:] =  [[[0.2, 0.8],
    #                      [0.9, 0.1]],
    #
    #                     [[0.9,  0.1],
    #                      [0.2,  0.8]]]
    for i in range(N):
        if i%2==0:
            T[i,:,:,:] =  [[[0.97, 0.03],
                     [0.03,  0.97]],

                    [[0.25, 0.75],
                     [0.03, 0.97]]]
        else:
            T[i,:,:,:] =  [[[0.96, 0.04],
                         [0.01, 0.99]],

                        [[0.23,  0.77],
                         [0.01,  0.99]]]

    return T


def generateTmatrix(N, responsive_patient_fraction=0.4,
                    range_pass_00=(0.7, 1.0), range_pass_11=(0.5, 0.9),
                    range_act_g_00=(0, 0.3), range_act_g_11=(0.8, 1.0),
                    range_act_b_00=(0.6, 0.9), range_act_b_11=(0.7, 1.0)):
    # p_act01 < p01/(p01+p10)

    """
    Generates a Nx2x2x2 T matrix indexed as: T[patient_number][action][current_state][next_state]
    action=0 denotes passive action, a=1 is active action
    State 0 denotes NA and state 1 denotes A
    """

    T = np.zeros((N, 2, 2, 2))
    # Passive action transition probabilities
    p_pass_00 = np.random.uniform(low=range_pass_00[0], high=range_pass_00[1], size=N)
    p_pass_11 = np.random.uniform(low=range_pass_11[0], high=range_pass_11[1], size=N)

    # Active action transition probabilities
    # responsive_patient_fraction=0.4
    p_act_00 = np.zeros(N)
    p_act_11 = np.zeros(N)
    for i in range(N):
        if np.random.binomial(1, responsive_patient_fraction) == 1:
            # Patient responds well to call
            p_act_00[i] = np.random.uniform(low=range_act_g_00[0], high=range_act_g_00[1])
            p_act_11[i] = np.random.uniform(low=range_act_g_11[0], high=range_act_g_11[1])
        else:
            # Patient doesn't respond well to call
            p_act_00[i] = np.random.uniform(low=range_act_b_00[0], high=range_act_b_00[1])
            p_act_11[i] = np.random.uniform(low=range_act_b_11[0], high=range_act_b_11[1])

    for i in range(N):
        T[i, 0] = np.array([[p_pass_00[i], 1 - p_pass_00[i]], [1 - p_pass_11[i], p_pass_11[i]]])
        T[i, 1] = np.array([[p_act_00[i], 1 - p_act_00[i]], [1 - p_act_11[i], p_act_11[i]]])

    # print (T[:20])
    return T

def generateTmatrixReal_(N, responsive_patient_fraction=0.5,
                    range_pass_g_00=(0, 0.2), range_pass_g_11=(0.87, 1.0),
                    range_pass_b_00=(0.3, 0.6), range_pass_b_11=(0.4, 0.7),
                    # range_pass_00=(0.35,0.55), range_pass_11=(0.57, 0.8),
                    range_act_g_00=(0, 0.1), range_act_g_11=(0.95, 1.0),
                    range_act_b_00=(0.001, 0.4), range_act_b_11=(0.6, 0.999)):
    # p_act01 < p01/(p01+p10)

    """
    Generates a Nx2x2x2 T matrix indexed as: T[patient_number][action][current_state][next_state]
    action=0 denotes passive action, a=1 is active action
    State 0 denotes NA and state 1 denotes A
    The real world prbability adherent in Pr(s0) = (0.025+0.023) Pr(s1)=0.951
    thus p_00^1 = 0.025*(0.0385)+0.951*0.0.0257
    p_11^1=0.951*0.9498
    Similarly can get
    for non-adherent is Pr(s0) = ( 0.495+0.094) Pr(s1)= 0.411
    """

    T = np.zeros((N, 2, 2, 2))
    # Passive action transition probabilities
    # p_pass_00 = np.random.uniform(low=range_pass_00[0], high=range_pass_00[1], size=N)
    # p_pass_11 = np.random.uniform(low=range_pass_11[0], high=range_pass_11[1], size=N)

    # Active action transition probabilities
    # responsive_patient_fraction=0.4
    p_act_00 = np.zeros(N)
    p_act_11 = np.zeros(N)
    for i in range(N):
        if np.random.binomial(1, responsive_patient_fraction) == 1:
            # Patient responds well to call
            p_pass_00 = np.random.uniform(low=range_pass_g_00[0], high=range_pass_g_00[1], size=N)
            p_pass_11 = np.random.uniform(low=range_pass_g_11[0], high=range_pass_g_11[1], size=N)

            p_act_00[i] = np.random.uniform(low=range_act_g_00[0], high=range_act_g_00[1])
            p_act_11[i] = np.random.uniform(low=range_act_g_11[0], high=range_act_g_11[1])
        else:
            # Patient doesn't respond well to call
            p_pass_00 = np.random.uniform(low=range_pass_b_00[0], high=range_pass_b_00[1], size=N)
            p_pass_11 = np.random.uniform(low=range_pass_b_11[0], high=range_pass_b_11[1], size=N)

            p_act_00[i] = np.random.uniform(low=range_act_b_00[0], high=range_act_b_00[1])
            p_act_11[i] = np.random.uniform(low=range_act_b_11[0], high=range_act_b_11[1])

    for i in range(N):


        T[i, 0] = np.array([[p_pass_00[i], 1 - p_pass_00[i]], [1 - p_pass_11[i], p_pass_11[i]]])
        T[i, 1] = np.array([[p_act_00[i], 1 - p_act_00[i]], [1 - p_act_11[i], p_act_11[i]]])

    # print (T[:20])
    return T