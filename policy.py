import numpy as np
import pandas as pd
import time
# import pomdp

from itertools import combinations


import os
import argparse
import tqdm

from utils import returnKGreatestIndices, generateRandomTmatrix, verify_T_matrix, epsilon_clip, barPlot, soft_max, prob_cal, soft_max_stable

from whittle import whittleIndex

from generateTmatrix import generateTmatrixReal, generateTmatrixBadf, generateDesignedTmatrix, generateTmatrix, generateTmatrixReal_

OPT_SIZE_LIMIT = 8
FULL_OBS = True
CONST = 9
NUM_SEED = 96

import random
random.seed( 10 )
def takeAction(adherence, current_adherence, belief, actions, T, random_stream, fairness_constraints, fairness_constraint=10, T_hat=None):
    """
    belief update after action
    """
    N = len(current_adherence)

    ###### Get next adhrence (ground truth)
    # Use the ground truth T here
    next_adherence = np.zeros(current_adherence.shape)
    for i in range(N):
        current_state = int(current_adherence[i])

        next_state = random_stream.binomial(1, T[i, int(actions[i]), current_state, 1])

        next_adherence[i] = next_state

    ##### Update belief vector
    # Remember to use T_hat here
    for i in range(N):
        if FULL_OBS:
            belief[i] = current_adherence[i]
        # belief = Prob(A)*Prob(A-->A) + Prob(NA)*Prob(NA-->A)
        else:
            if int(actions[i]) == 0:
                belief[i] = belief[i] * T_hat[i][0][1][1] + (1 - belief[i]) * (T_hat[i][0][0][1])

            elif int(actions[i]) == 1:
                belief[i] = current_adherence[i] * T_hat[i][1][1][1] + (1 - current_adherence[i]) * (T_hat[i][1][0][1])
                #   This relies on the assumption that on being called at least yesterday's
                #   adherence is perfectly observable. If not replace current_adherence[i] by belief[i]

    ##### Record observation
    observations = np.zeros(N)
    for i in range(N):
        if FULL_OBS:
            observations[i] = current_adherence[i]
        else:
            if actions[i] == 0:
                observations[i] = None
            else:
                observations[i] = current_adherence[i]

    for i in range(N):
        if actions[i] == 0:
            fairness_constraints[i] -= 1
        else:
            fairness_constraints[i] = fairness_constraint

    return next_adherence, belief, observations, fairness_constraints


def getActions(N, k, fairness_constraint, fairness_constraints, Q_tables, V_tables, alpha, epsilon_t, time_steps=None, belief=None, T_hat=None,policy_option=0,
               current_node=None, policy_graph_dict=None,
               days_since_called=None,last_observed_state=None,w=None,w_new=None,newWhittle=True,
               adherence_oracle=None, days_remaining=None, current_t=None,
               observations=None, adherence=None, T=None, verbose=False):
    """
    0: never call
    1: Call all patients everyday
    2: Randomly pick k patients to call
    5: whittle
    """
    if policy_option==0:
        ################## Nobody
        return np.zeros(N)

    elif policy_option==1:
        ################## Everybody
        return np.ones(N)

    elif policy_option==2:
        ################## Random
        # Randomly pick k arms out N arms
        random_call_indices=np.random.choice(N,size=k, replace=False)
        return np.array([1 if i in random_call_indices else 0 for i in range(N)])

    elif policy_option == 3:
        ################## Myopic policy
        actions = np.zeros(N)
        myopic_rewards = np.zeros(N)
        for i in range(N):
            b = belief[i]  # Patient is adhering today with probability b

            b_next_nocall = b * (T_hat[i][0][1][1]) + (1 - b) * (T_hat[i][0][0][1])

            b_next_call = b * (T_hat[i][1][1][1]) + (1 - b) * (T_hat[i][1][0][1])

            myopic_rewards[i] = b_next_call - b_next_nocall
            # Myopic reward can be thought of as: Prob(A)*1 + Prob(NA)*0 = b

        # Pick the k greatest values from the array myopic_rewards
        patients_to_call = returnKGreatestIndices(fairness_constraints, myopic_rewards, k)

        for patient in patients_to_call:
            actions[patient] = 1

        return actions

    elif policy_option == 5:
        ################## Whittle Index policy
        # Initialize if inputs not given
        if days_since_called.any() == None:
            days_since_called = np.zeros(N)  # Initialize to 0 days since last called (means nothing much)

        if last_observed_state.any() == None:
            last_observed_state = np.ones(N)  # Initialize to all patients found adhering last

        actions = actions = np.zeros(N)

        whittle_indices = [w[patient][int(last_observed_state[patient])][int(days_since_called[patient])] for patient in
                           range(N)]

        patients_to_call = returnKGreatestIndices(fairness_constraints, whittle_indices, k)

        for patient in patients_to_call:
            actions[patient] = 1

        return actions

    elif policy_option == 19:
        # whittleIndex based Q learning

        # Initialize if inputs not given
        if days_since_called.any() == None:
            days_since_called = np.zeros(N)  # Initialize to 0 days since last called (means nothing much)

        if last_observed_state.any() == None:
            last_observed_state = np.ones(N)  # Initialize to all patients found adhering last

        if np.random.random() < epsilon_t:
            random_call_indices = np.random.choice(N, size=k, replace=False)
            return np.array([1 if i in random_call_indices else 0 for i in range(N)])
        else:
            whittle_indices = np.ones(N)
            actions = np.zeros(N)
            for j in range(N):
                # time = fairness_constraints[j]
                # if time<= 0:
                #     time = 0
                # period = fairness_constraint-time
                period = days_since_called[j].astype(int)
                if period >= fairness_constraint:
                    period = fairness_constraint-1
                state = last_observed_state[j].astype(int)
                # get the whittleindex based Q values Lambda(X(t))
                whittle_indices[j] = Q_tables[j][period][state][1] - Q_tables[j][period][state][0]
            patients_to_call = returnKGreatestIndices(fairness_constraints, whittle_indices, k)
            for patient in patients_to_call:
                actions[patient] = 1

            return actions

    elif policy_option == 6:
        # whittleIndex based Softmax

        # Initialize if inputs not given
        if days_since_called.any() == None:
            days_since_called = np.zeros(N)  # Initialize to 0 days since last called (means nothing much)

        if last_observed_state.any() == None:
            last_observed_state = np.ones(N)  # Initialize to all patients found adhering last

        diff = np.ones(N)
        actions = np.zeros(N)
        period = time_steps
        for j in range(N):
            state = last_observed_state[j].astype(int)
            V_active = T_hat[j][1][state][0]*V_tables[j][period][0] + T_hat[j][1][state][1]*V_tables[j][period][1]
            V_passive = T_hat[j][0][state][0]*V_tables[j][period][0] + T_hat[j][0][state][1]*V_tables[j][period][1]
            diff[j] = V_active- V_passive

            # tricks to augment the difference
            diff[j] = diff[j]*CONST
        prob = soft_max(diff)

        # tricks to augment the difference
        # tmp_prob = prob_cal(prob,k)
        # tmp_prob = prob/np.sum(prob)

        # print('prob of being choosed:', prob)
        # sample actions
        random_call_indices = np.random.choice(N, size=k, replace=False, p=prob)
        # random_call_indices = np.random.choice(N, size=k, replace=False, p=tmp_prob)
        actions = np.array([1 if i in random_call_indices else 0 for i in range(N)])
        # print(actions-prob)

        # deterministically choose
        # actions = np.zeros(N)

        # patients_to_call = returnKGreatestIndices(fairness_constraints, prob, k)

        # for patient in patients_to_call:
        #     actions[patient] = 1

        return actions, prob
    elif policy_option == 7:
        ################## Myopic policy
        actions = np.zeros(N)
        myopic_rewards = np.zeros(N)
        for i in range(N):
            state = last_observed_state[i].astype(int)

            myopic_rewards[i] = T_hat[i][1][state][1] - T_hat[i][0][state][1]
            # Myopic reward can be thought of as: Prob(A)*1 + Prob(NA)*0 = b

        # Pick the k greatest values from the array myopic_rewards
        prob = soft_max(myopic_rewards)
        new_prob = prob_cal(prob,k)
        new_prob = new_prob/np.sum(new_prob)
        # print('prob of being choosed:', prob)
        random_call_indices = np.random.choice(N, size=k, replace=False, p=new_prob)
        actions = np.array([1 if i in random_call_indices else 0 for i in range(N)])

        return actions
    elif policy_option == 8:
        ################## Myopic policy
        actions = np.zeros(N)
        myopic_rewards = np.zeros(N)
        for i in range(N):
            state = last_observed_state[i].astype(int)

            myopic_rewards[i] = T_hat[i][1][state][1] - T_hat[i][0][state][1]
            # Myopic reward can be thought of as: Prob(A)*1 + Prob(NA)*0 = b

        # Pick the k greatest values from the array myopic_rewards
        patients_to_call = returnKGreatestIndices(fairness_constraints, myopic_rewards, k)

        for patient in patients_to_call:
            actions[patient] = 1

        return actions

def learnTmatrixFromObservations(observations, actions, random_stream):
    '''
    observations and actions are L+1 and L-sized matrices with:
        Observations: [o0, o1,...oL] with each entry being 0=NA;  1=A
        Actions:      [a1, a2,...aL] with each entry being 0=NoCall; 1=Called
    '''
    T = np.zeros((2, 2, 2))
    p_pass_01, p_pass_11, p_act_01, p_act_11 = sorted(random_stream.uniform(size=4))
    l = len(actions)
    vals, counts = np.unique(list(zip(observations[:l], actions, observations[1:])), axis=0, return_counts=True)

    freq = np.zeros((2, 2, 2))

    for i, item in enumerate(vals):
        freq[int(item[0]), int(item[1]), int(item[2])] = counts[i]

    if (freq[0, 0, 0] + freq[0, 0, 1]) > 0:
        p_pass_01 = freq[0, 0, 1] / (freq[0, 0, 0] + freq[0, 0, 1])

    if (freq[1, 0, 0] + freq[1, 0, 1]) > 0:
        p_pass_11 = freq[1, 0, 1] / (freq[1, 0, 0] + freq[1, 0, 1])

    if (freq[0, 1, 0] + freq[0, 1, 1]) > 0:
        p_act_01 = freq[0, 1, 1] / (freq[0, 1, 0] + freq[0, 1, 1])

    if (freq[1, 1, 0] + freq[1, 1, 1]) > 0:
        p_act_11 = freq[1, 1, 1] / (freq[1, 1, 0] + freq[1, 1, 1])

    T[0] = np.array([[1 - p_pass_01, p_pass_01], [1 - p_pass_11, p_pass_11]])
    T[1] = np.array([[1 - p_act_01, p_act_01], [1 - p_act_11, p_act_11]])

    return T


def update_counts(adherence, actions, last_called, current_round, counts, buffer_length=0, get_last_call_transition_flag=False):
	if buffer_length == 0:
		buffer_length = 100000000

	# Buffer is how much patient "remembers" which doesn't include today.
	# so add 1 to the buffer_length to make code cleaner below, i.e. adding
	# 1 makes the buffer include today.
	buffer_length+=1

	patients_called = [i for i,a in enumerate(actions) if a==1]

	for i in patients_called:
		info_packet = adherence[i, last_called[i]:current_round+1].astype(int)

		curr = None
		prev = None

		# if it doesn't fit in the buffer cut it, but conditionally add the t1
		# remaining adds will be to t0
		if info_packet.shape[0] > buffer_length:

			if get_last_call_transition_flag:
				prev = info_packet[0]
				curr = info_packet[1]
				counts[i, 1, prev, curr] += 1

			info_packet = info_packet[-buffer_length:]
			prev = info_packet[0]
			curr = info_packet[1]
			counts[i, 0, prev, curr] += 1
			prev = curr

		# Else first add will be to t1
		else:
			prev = info_packet[0]
			curr = info_packet[1]
			counts[i, 1, prev, curr] += 1
			prev = curr

		# The rest is about T0
		for j in range(2, len(info_packet)):
			curr = info_packet[j]
			counts[i, 0, prev, curr] += 1
			prev = curr

		# record that we called this patient
		last_called[i] = current_round


def thompson_sampling(N, priors, counts, random_stream):

    T_hat = np.zeros((N,2,2,2))
    for i in range(N):
        for j in range(T_hat.shape[1]):
            for k in range(T_hat.shape[2]):
                params = priors[i, j, k, :] + counts[i, j, k, :]
                T_hat[i, j, k, :] = random_stream.dirichlet(params)
    return T_hat


def thompson_sampling_constrained(N, priors, counts, random_stream):

    T_hat = np.zeros((N,2,2,2))
    for i in range(N):
    	# While sampled T_hat is not valid or has not been sampled yet...
    	while (not verify_T_matrix(T_hat[i]) or T_hat[i].sum() == 0):
	        for j in range(T_hat.shape[1]):
	            for k in range(T_hat.shape[2]):
	                params = priors[i, j, k, :] + counts[i, j, k, :]
	                T_hat[i, j, k, :] = random_stream.dirichlet(params)
    return T_hat


def simulateAdherence(N,L,T,k, policy_option, fairness_constraints, alpha, gamma, Q_tables, start_node=None, policy_graph_dict=None,
                        obs_space=None, action_logs={}, cum_adherence=None,
                        new_whittle=True, online=True,
                        seedbase=None, savestring='trial', epsilon=0.0, learning_mode=False,
                        world_random_seed=None, learning_random_seed=None, verbose=False,
                        buffer_length=0, get_last_call_transition_flag=False, fairness_constraint=10, file_root=None):
    learning_random_stream = np.random.RandomState()
    if learning_mode > 0:
        learning_random_stream.seed(learning_random_seed)

    world_random_stream = np.random.RandomState()
    world_random_stream.seed(world_random_seed)

    T_hat = None
    if learning_mode == 2:
        T_hat = generateRandomTmatrix(N, random_stream=learning_random_stream)
    priors = np.ones((N,2,2,2))
    counts = np.zeros((N,2,2,2))
    last_called = np.zeros(N).astype(int)

    # if learning_mode == 4:
    # 	T_hat = computeAverageTmatrixFromData(N, file_root=file_root)

    adherence=np.zeros((N,L))
    actions_record=np.zeros((N, L-1))

    # record the penalty
    penalty = np.zeros((N,L))

    if action_logs is not None:
        action_logs[policy_option] = []

    adherence[:,0]=np.ones(N)
    belief=np.ones(N)

    penalty[:, 0]=np.zeros(N)
    current_node = None

    w=None
    w_new=None

    if (policy_option==5 and (not online) and (not learning_mode)):
        # Pre-compute only if policy is whittle index AND it is neither online nor learning case.
        print("Pre-computing whittle index for offline, no-learning mode")
        # Pre-compute whittle index for patients
        w = np.zeros((N, 2, L))
        w_new = np.zeros((N, 2, L))  # right now, w_new does not get used in takeAction() even though it's passed in.
        for patient in range(N):
            if policy_option == 5:
                w[patient, 1, :], w[patient, 0, :] = whittleIndex(T[patient], L=L)
            # if policy_option == 18:
            #     w[patient, 0, :], w[patient, 1, :] = whittleIndex(T[patient], L=L)

    # Keep track of days since called and last observed state
    days_since_called = np.zeros(N)  # Initialize to 0 days since called
    last_observed_state = np.ones(N)

    fairness_constraints = fairness_constraints

    #######  Run simulation #######
    print('Running simulation w/ policy: %s' % policy_option)
    # make array of nan to initialize observations
    observations = np.full(N, np.nan)
    learning_modes = ['no_learning', 'Thompson sampling', 'e-greedy', 'Constrained TS', 'Naive Mean']
    print("Learning mode:", learning_modes[learning_mode])

    epsilon_schedule = [epsilon]*(L-1) # always explore with epsilon
    if epsilon == 0.0: # else anneal epsilon from 1 to 0.0.
        # power = np.log(y)/np.log(1-x)
        # where y = desired epsilon when we are x% of the way through treatment
        # so if we want epsilon to be 0.25 by the time we are 25% of the way through treatment
        # we get: np.log(0.25)/np.log(1-0.25) = 4.818841679306418
        power = 4.818841679306418
        epsilon_schedule = np.linspace(1,0.00,L)**power
        # Note that we never have epsilon 0 since we never access the last element.

    actions_fairness_log = np.zeros(args.num_patients)
    for t in tqdm.tqdm(range(1, L)):
        '''
        Learning T_hat from simulation so far
        '''
        epsilon_t = N/(N+t)
        days_remaining = L-t
        if learning_mode==0:
            T_hat=T
        elif learning_mode == 1:
            # Thompson sampling
            T_hat = thompson_sampling(N, priors, counts, random_stream=learning_random_stream)
        elif learning_mode==2 and t>2:
            # Epsilon-Greedy
            for patient_number, action in enumerate(actions):# Note that actions here is still the previous day's action record
                if action==1:
                    T_hat[patient_number]=learnTmatrixFromObservations(adherence[patient_number, :t-1],
                        actions_record[patient_number, 1:(t-1)], random_stream=learning_random_stream)
        elif learning_mode == 3:
            # Thompson sampling
            T_hat = thompson_sampling_constrained(N, priors, counts, random_stream=learning_random_stream)

        elif learning_mode == 4:
            # Naive mean
            pass

        EPSILON_CLIP=0.0005
        T_hat= epsilon_clip(T_hat, EPSILON_CLIP)

        if online or learning_mode:
            # If neither online nor learning, then just work with pre-computed whittle indices, w and w_new.
            w=np.zeros((N, 2, L))
            w_new=np.zeros((N, 2, L))

            if policy_option == 5:

                for patient in range(N):
                    limits = [0, 0]
                    limits[int(last_observed_state[patient])] = days_since_called[patient] + 1
                    w[patient, 1, :], w[patient, 0, :] = whittleIndex(T_hat[patient], L=L, limit_a=limits[1],
                                                                      limit_na=limits[0])
            # other fast whittle index calculation
            if policy_option == 16:
                pass

        #### Epsilon greedy part

        if learning_mode == 2 and (policy_option not in NON_EPSILON_POLICIES):  # epsilon greedy
            if learning_random_stream.binomial(1, epsilon_schedule[t]) == 0:  # Exploitation
                if policy_option != 6:
                    actions = getActions(N, k, fairness_constraint, fairness_constraints, Q_tables, V_tables, alpha, epsilon_t, time_steps=t, epolicy_option=policy_option, belief=belief, T_hat=T_hat,
                                         current_node=current_node, policy_graph_dict=policy_graph_dict,
                                         days_since_called=days_since_called,
                                         last_observed_state=last_observed_state, w=w, w_new=w_new, current_t=t,
                                         adherence_oracle=adherence[:, t - 1].squeeze(), days_remaining=days_remaining,
                                         observations=observations, adherence=adherence[:, t - 1], T=T,
                                         verbose=verbose)
                else:
                    actions, prob = getActions(N, k, fairness_constraint, fairness_constraints, Q_tables, V_tables, alpha, epsilon_t, time_steps=t, epolicy_option=policy_option, belief=belief, T_hat=T_hat,
                                         current_node=current_node, policy_graph_dict=policy_graph_dict,
                                         days_since_called=days_since_called,
                                         last_observed_state=last_observed_state, w=w, w_new=w_new, current_t=t,
                                         adherence_oracle=adherence[:, t - 1].squeeze(), days_remaining=days_remaining,
                                         observations=observations, adherence=adherence[:, t - 1], T=T,
                                         verbose=verbose)
            else:  # Exploration
                if policy_option != 6:
                    actions=getActions(N, k, fairness_constraint, fairness_constraints, Q_tables, V_tables,alpha, epsilon_t, time_steps=t, policy_option=2, belief=belief, T_hat=T_hat,
                                       current_node=current_node, policy_graph_dict = policy_graph_dict,
                                       days_since_called=days_since_called,
                                       last_observed_state=last_observed_state, w=w,w_new=w_new,current_t=t,
                                       adherence_oracle=adherence[:,t-1].squeeze(), days_remaining=days_remaining,
                                       observations=observations, adherence=adherence[:, t-1], T=T,
                                       verbose=verbose)
                else:
                    actions,prob=getActions(N, k, fairness_constraint, fairness_constraints, Q_tables, V_tables,alpha, epsilon_t, time_steps=t, policy_option=2, belief=belief, T_hat=T_hat,
                                       current_node=current_node, policy_graph_dict = policy_graph_dict,
                                       days_since_called=days_since_called,
                                       last_observed_state=last_observed_state, w=w,w_new=w_new,current_t=t,
                                       adherence_oracle=adherence[:,t-1].squeeze(), days_remaining=days_remaining,
                                       observations=observations, adherence=adherence[:, t-1], T=T,
                                       verbose=verbose)

        else: # Normal process
            if policy_option != 6:
                actions=getActions(N, k, fairness_constraint, fairness_constraints, Q_tables, V_tables, alpha, epsilon_t, time_steps=t, policy_option=policy_option, belief=belief, T_hat=T_hat,
                   current_node=current_node, policy_graph_dict = policy_graph_dict,
                   days_since_called=days_since_called,
                   last_observed_state=last_observed_state, w=w,w_new=w_new, current_t=t,
                   adherence_oracle=adherence[:,t-1].squeeze(), days_remaining=days_remaining,
                   observations=observations, adherence=adherence[:, t-1], T=T,
                   verbose=verbose)
            else:
                actions, prob =getActions(N, k, fairness_constraint, fairness_constraints, Q_tables, V_tables, alpha, epsilon_t, time_steps=t, policy_option=policy_option, belief=belief, T_hat=T_hat,
                   current_node=current_node, policy_graph_dict = policy_graph_dict,
                   days_since_called=days_since_called,
                   last_observed_state=last_observed_state, w=w,w_new=w_new, current_t=t,
                   adherence_oracle=adherence[:,t-1].squeeze(), days_remaining=days_remaining,
                   observations=observations, adherence=adherence[:, t-1], T=T,
                   verbose=verbose)

        actions_record[:, t - 1] = actions

        for i in range(args.num_patients):
            if actions[i]==1:
                actions_fairness_log[i] += 1
        if t==L-1:
            tmp_a = []

            for i in range(len(actions_fairness_log)):
                tmp_a.append(actions_fairness_log[i])
            print('action fairness', tmp_a)
            pass


        if action_logs is not None:
            action_logs[policy_option].append(actions.astype(int))

        adherence[:,t], belief, observations, fairness_constraints=takeAction(adherence, adherence[:,t-1].squeeze(), belief, actions,T, random_stream=world_random_stream,fairness_constraints=fairness_constraints, fairness_constraint=fairness_constraint, T_hat=T_hat)

        penalty[:, t] = [0 if fairness_constraints[i]>=0 else 1 for i in range(N)]

        # update counts
        # get all information about a patient since the last time we called
        # the transition (last_called[i], last_called[i]+1) will be the only info we get about a T1 matrix
        # all others will update our info about T0
        update_counts(adherence, actions, last_called, t, counts, buffer_length=buffer_length, get_last_call_transition_flag=get_last_call_transition_flag)

        # Update the whittleIndex baseed Q value
        if policy_option == 19:
            states_new_observed = adherence[:, t]
            for i in range(N):
                a = actions[i].astype(int)
                days_since_last_called = days_since_called[i].astype(int)
                if days_since_last_called >= fairness_constraint:
                    days_since_last_called = fairness_constraint - 1
                state_last_observed = last_observed_state[i].astype(int)
                state_new_observed = states_new_observed[i].astype(int)
                q_target = adherence[i,t] + gamma * Q_tables[i][days_since_last_called][state_new_observed].max()
                q_predict = Q_tables[i][days_since_last_called][state_last_observed][a]
                Q_tables[i][days_since_last_called][state_last_observed][a] += alpha * (q_target - q_predict)
        if policy_option == 6:
            probs = prob_cal(prob, k)
            states_new_observed = adherence[:, t]
            for i in range(N):
                a = actions[i].astype(int)
                # days_since_last_called = days_since_called[i].astype(int)
                horizon = t - 1
                state_last_observed = last_observed_state[i].astype(int)
                state_new_observed = states_new_observed[i].astype(int)
                # q_target = adherence[i,t] + gamma * Q_tables[i][days_since_last_called][state_new_observed].max()
                # q_predict = Q_tables[i][days_since_last_called][state_last_observed][a]
                # Q_tables[i][days_since_last_called][state_last_observed][a] += alpha * (q_target - q_predict)
                if t < L-1:
                    Q_active = T_hat[i][1][state_last_observed][0] * (state_last_observed + V_tables[i][horizon+1][0]) + T_hat[i][1][state_last_observed][1] * (state_last_observed + V_tables[i][horizon+1][1])
                    Q_passive = T_hat[i][0][state_last_observed][0] * (state_last_observed + V_tables[i][horizon+1][0]) + T_hat[i][0][state_last_observed][1] * (state_last_observed + V_tables[i][horizon+1][1])
                    V_tables[i][horizon][state_last_observed] = probs[i] * Q_active + (1-probs[i]) *Q_passive
                else:
                    V_tables[i][horizon][state_last_observed] = state_last_observed


        ###### Update last observed state and last called matrix:
        for i in range(N):
            if actions[i] == 0:
                days_since_called[i] += 1
            else:
                days_since_called[i] = 0
                last_observed_state[i] = observations[i]

    if cum_adherence is not None:
        cum_adherence[policy_option] = np.cumsum(adherence.sum(axis=0))

    return adherence, penalty


if __name__=="__main__":
    """
    0: never call    1: Call all patients everyday     2: Randomly pick k patients to call
    3: Myopic policy    4: pomdp  5: whittle  
    """
    parser = argparse.ArgumentParser(description='Run simulations')
    parser.add_argument('-n', '--num_patients', default=100, type=int, help='Number of Patients')
    parser.add_argument('-eps', '--episode', default=80, type=int, help='Number of episodes')
    parser.add_argument('-k', '--num_calls_per_day', default=10, type=float, help='Number of calls per day')
    parser.add_argument('-l', '--simulation_length', default=70, type=int, help='Number of days to run simulation')
    parser.add_argument('-N', '--num_trials', default=5, type=int, help='Number of trials to run')
    parser.add_argument('-d', '--data', default='real',
                        choices=['real', 'simulated', 'full_random', 'unit_test', 'myopic_fail', 'demo', 'uniform', 'designed'],
                        type=str, help='Method for generating transition probabilities')
    parser.add_argument('-s', '--seed_base', type=int, help='Base for the random seed')
    parser.add_argument('-ws', '--world_seed_base', default=None, type=int, help='Base for the random seed')
    parser.add_argument('-ls', '--learning_seed_base', default=None, type=int, help='Base for the random seed')
    parser.add_argument('-p', '--num_processes', default=4, type=int, help='Number of cores for parallelization')
    parser.add_argument('-f', '--file_root', default='./..', type=str,
                        help='Root dir for experiment (should be the dir containing this script)')
    parser.add_argument('-pc', '--policy', default=-1, type=int, help='policy to run, default is all policies')
    parser.add_argument('-res', '--results_file', default='answer', type=str, help='adherence numpy matrix file name')
    parser.add_argument('-tr', '--trial_number', default=None, type=int, help='Trial number')
    parser.add_argument('-sv', '--save_string', default='', type=str,
                        help='special string to include in saved file name')
    parser.add_argument('-badf', '--bad_fraction', default=0.4, type=float, help='fraction of non-responsive patients')
    parser.add_argument('-thrf_perc', '--threshopt_percentage', default=None, type=int,
                        help='% of threshold optimal patients in data')
    parser.add_argument('-beta', '--beta', default=0.5, type=float,
                        help='beta used in quick check for determining threshold optimal fraction')
    parser.add_argument('-ep', '--epsilon', default=0, type=float, help='espilon value for epsilon greedy')
    parser.add_argument('-lr', '--learning_option', default=0, choices=[0, 1, 2, 3, 4], type=int,
                        help='0: No Learning (Ground truth known)\n1: Thompson Sampling\n2 Epsilon Greedy\n3 Constrained TS\n4 Naive average baseline')
    parser.add_argument('-v', '--verbose', default=False, type=bool)
    parser.add_argument('-o', '--online', default=0, type=int, help='0: offline, 1: online')
    parser.add_argument('-kp', '--k_percentage', default=None, type=int, help='100* k/N ')
    parser.add_argument('-slurm', '--slurm_array_id', default=-1, type=int,
                        help='Unique identifier for slurm array id/ encoding set of parameters')
    parser.add_argument('-sh1', '--shift1', default=0.05, type=float, help='shift 1 variable ')
    parser.add_argument('-sh2', '--shift2', default=0.05, type=float, help='shift 2 variable ')
    parser.add_argument('-sh3', '--shift3', default=0.05, type=float, help='shift 3 variable ')
    parser.add_argument('-sh4', '--shift4', default=0.05, type=float, help='shift 4 variable ')
    parser.add_argument('-bl', '--buffer_length', default=0, type=int,
                        help='If using Thompson Sampling, max number of most recent days of adherence you learn with an arm pull')
    parser.add_argument('-t1f', '--get_last_call_transition_flag', default=0, type=int,
                        help='If using Thompson Sampling, whether or not you learn the T1 transition regardless of buffer_length with an arm pull')
    parser.add_argument('-fc', '--fairness_constraint', default=1000, type=int, help='fairness constraint for each patient')
    parser.add_argument('-cfc', '--constant_fairness_constraint', default=True, type = bool, help='If True, means that all the arms are initialized with same constant fairness constraint')
    parser.add_argument('-alp', '--alpha', default=0.5, type=float, help='alpha parameter')
    parser.add_argument('-gm', '--gamma', default=0.95, type=float, help='gamma parameter')

    args = parser.parse_args()
    NON_EPSILON_POLICIES = [0, 1, 2, 5]

    if args.slurm_array_id >= 0:
        '''
        Changing tr: 0-49, policy: {10,14}, N:{10,20,100,200,500,1000,2000}
        '''
        slurm_trial_nums = [i for i in range(50)]
        slurm_policies = [10, 14]
        slurm_N = [200, 500, 1000, 2000]
        # slurm_th_fracs=[0,10,20,30,40,50,60,70,80,90,100]

        args.trial_number=args.slurm_array_id%len(slurm_trial_nums)
        args.policy= slurm_policies[int(args.slurm_array_id//len(slurm_trial_nums))%len(slurm_policies)]
        args.num_patients=slurm_N[int(args.slurm_array_id//(len(slurm_trial_nums)*len(slurm_policies)))%len(slurm_N)]
        #args.threshopt_percentage=slurm_th_fracs[int(args.slurm_array_id//(len(slurm_trial_nums)*len(slurm_policies)*len(slurm_N)))%len(slurm_th_fracs)]
        #args.save_string+=("_threshopt_frac_"+str(args.threshopt_percentage))


    ##### File root
    if args.file_root == '.':
        args.file_root = os.getcwd()
    ##### k
    args.num_calls_per_day = int(args.num_calls_per_day)
    if args.k_percentage is not None:
        args.num_calls_per_day = int((args.k_percentage / 100 * args.num_patients))  # This rounds down, good.

    ##### Save special name
    if args.save_string=='':
        args.save_string=str(time.ctime().replace(' ', '_').replace(':','_'))
    else:
        args.save_string=args.save_string

    ##### Policies to run
    if args.policy < 0:
        # policies = [0,1,2,3,5, 10, 14] # all relevant policies
        policies = [0, 1, 2, 5]  # RUN FAST POLICIES ONLY
        policies = [0, 1, 2, 5, 6,7,8]
        # policies = [6]

    else:
        policies = [args.policy]

    ##### Seed = seed_base + trial_number
    if args.trial_number is not None:
        args.num_trials=1
        add_to_seed_for_specific_trial=args.trial_number
    else:
        add_to_seed_for_specific_trial=0
    first_seedbase=np.random.randint(0, high=100000)
    if args.seed_base is not None:
        first_seedbase = args.seed_base+add_to_seed_for_specific_trial

    first_world_seedbase=np.random.randint(0, high=100000)
    if args.world_seed_base is not None:
        first_world_seedbase = args.world_seed_base+add_to_seed_for_specific_trial

    first_learning_seedbase=np.random.randint(0, high=100000)
    if args.learning_seed_base is not None:
        first_learning_seedbase = args.learning_seed_base+add_to_seed_for_specific_trial

    ##### Other parameters
    N=args.num_patients
    # N = 10
    L=args.simulation_length
    k=args.num_calls_per_day
    # k = 4
    fairness_constraint = args.fairness_constraint
    savestring=args.save_string
    N_TRIALS=args.num_trials
    LEARNING_MODE=args.learning_option
    #LEARNING_MODE='EpsilonGreedy'#'False'
    #LEARNING_MODE='Thompson'#'False'
    #LEARNING_MODE='False'
    alpha = args.alpha
    gamma = args.gamma

    num_actions = 2
    num_states = 2
    Q_tables = np.zeros((N, fairness_constraint, num_states, num_actions))

    if args.constant_fairness_constraint:
        fairness_constraints = np.ones(N)*fairness_constraint
    else:
        fairness_constraints = [np.random.randint(fairness_constraint) for i in range(N)]

    record_policy_actions=[3, 4, 5, 6, 11, 12, 13, 7, 8, 10, 14, 15,16, 17, 18, 19]
    record_policy_actions = [0,1,2,3,4,5,6,19]
    # size_limits: run policy if N< size_limit; ALso size_limit=-1 means all N are ok. Size_limit=0 means switched off.
    size_limits={ 0:None, 1:None, 2: None, 3: None, 4:OPT_SIZE_LIMIT, 5: None
                ,6: None ,    7:None,     8:None,  9:0,     10:None,
                11: None, 12: None, 13: None, 14: None, 15: None, 16:None, 17:None, 18:None, 19: None, 20: None}

    # policy names dict
    pname = {0: 'nobody', 1: 'everyday', 2: 'Random',
             3: 'Myopic', 4: 'optimal', 5: 'Threshold whittle',
             6: 'soft_max', 7: 'fair_myopic', 8: 'myopic',
             9: 'despot', 10: 'yundi', 11: 'naiveBelief',
             12: 'naiveReal', 13: 'naiveReal2', 14: 'oracle_MDP',
             15: 'Round Robin', 16: 'New_whittle', 17: 'FastWhittle',
             18: 'BuggyWhittle', 19: 'Q_learning'}

    adherences=[[] for i in range(len(pname))]
    penalties = [[] for i in range(len(pname))]
    adherence_matrices=[None for i in range(len(pname))]
    action_logs = {}
    cum_adherence = {}

    start=time.time()
    file_root=args.file_root

    Q_tables = np.zeros((N, args.simulation_length, num_states, num_actions))
    V_tables = np.zeros((N, args.simulation_length, num_states))
    for ep in range(args.episode):
        # Q_tables = np.zeros((N, fairness_constraint, num_states, num_actions))


        seedbase = first_seedbase + ep
        np.random.seed(seed=seedbase)

        world_seed_base = first_world_seedbase + ep
        learning_seed_base = first_learning_seedbase + ep

        # print (args.seed_base)
        print("Seed is", seedbase)
        # print (args.online)
        T = None
        if args.data == 'real':
            np.random.seed(NUM_SEED)
            T = generateTmatrixReal_(N, responsive_patient_fraction=1. - args.bad_fraction)
        #
        #     if args.threshopt_percentage is not None:
        #         T = generateTmatrixReal(N, file_root=args.file_root,
        #                                 thresh_opt_frac=(args.threshopt_percentage) / 100., beta=args.beta,
        #                                 quick_check=False)
        #     else:
        #         T = generateTmatrixReal(N, file_root=args.file_root,
        #                                 thresh_opt_frac=None, beta=args.beta, quick_check=False)
        if args.data == 'simulated':
            np.random.seed(NUM_SEED)
            # T = generateTmatrix(N, responsive_patient_fraction=1.- args.bad_fraction)
            T = generateTmatrixBadf(N, responsive_patient_fraction=1.- args.bad_fraction)
            # print('T:', T)
        elif args.data =='designed':
            np.random.seed(NUM_SEED)
            T = generateDesignedTmatrix(N)
            # print('T', T)
        np.random.seed(seed=seedbase)
            #N = 2
            #k=1

        for policy_option in policies:
            # print (policy_option)
            ############################ Policy # policy_option
            policy_start_time = time.time()
            if size_limits[policy_option]==None or size_limits[policy_option]>N:
                np.random.seed(seed=seedbase)
                if policy_option in record_policy_actions:
                    adherence_matrix, penalty_matrix = simulateAdherence(N, L, T, k, fairness_constraints = fairness_constraints,
                                                         alpha = alpha, gamma=gamma, Q_tables = Q_tables,
                                                         policy_option=policy_option, seedbase=seedbase,
                                                         action_logs=action_logs,
                                                         cum_adherence=cum_adherence, epsilon=args.epsilon,
                                                         learning_mode=LEARNING_MODE,
                                                         learning_random_seed=learning_seed_base,
                                                         world_random_seed=world_seed_base,
                                                         verbose=args.verbose, online=(args.online == 1),
                                                         buffer_length=args.buffer_length,
                                                         get_last_call_transition_flag=args.get_last_call_transition_flag,
                                                         fairness_constraint = fairness_constraint,
                                                         file_root=file_root)
                    print('!!!!!!!!!!!!!1policy option', policy_option)
                    adherence_matrices[policy_option] = adherence_matrix
                    # np.save(file_root+'/adherence_log/rebuttal/adherence_'+savestring+'_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s'%(N,k,L,policy_option,args.data,args.bad_fraction,seedbase, LEARNING_MODE), adherence_matrix)
                    np.save(
                        file_root + '/logs/adherence_log/adherence_' + savestring + '_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s_bl%s_t1f%s' % (
                        N, k, L, policy_option, args.data, args.bad_fraction, seedbase, LEARNING_MODE,
                        args.buffer_length, args.get_last_call_transition_flag), adherence_matrix)
                    # np.save(
                    #     file_root + '/logs/penalty_log/adherence_' + savestring + '_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s_bl%s_t1f%s' % (
                    #         N, k, L, policy_option, args.data, args.bad_fraction, seedbase, LEARNING_MODE,
                    #         args.buffer_length, args.get_last_call_transition_flag), penalty_matrix)

                    # adherences[policy_option].append(np.mean(np.sum(adherence_matrix, axis=1)))
                    # penalties[policy_option].append(np.mean(np.sum(penalty_matrix, axis=1)))

                    if policy_option ==66 :
                        if ep>L:
                            adherences[policy_option].append(np.mean(np.sum(adherence_matrix, axis=1)))
                            penalties[policy_option].append(np.mean(np.sum(penalty_matrix, axis=1)))
                    else:
                        adherences[policy_option].append(np.mean(np.sum(adherence_matrix, axis=1)))
                        penalties[policy_option].append(np.mean(np.sum(penalty_matrix, axis=1)))
                else:
                    if args.verbose:
                        print(learning_seed_base, 'LRSEED\n\n\n\n\n')
                        print(world_seed_base, 'LRSEED\n\n\n\n\n')

                    adherence_matrix, penalty_matrix = simulateAdherence(N, L, T, k, fairness_constraints=fairness_constraints, alpha=alpha,
                                                         gamma=gamma, Q_tables=Q_tables, policy_option=policy_option, seedbase=seedbase,
                                                         learning_mode=LEARNING_MODE,
                                                         learning_random_seed=learning_seed_base,
                                                         world_random_seed=world_seed_base,
                                                         verbose=args.verbose, online=(args.online == 1),
                                                         buffer_length=args.buffer_length,
                                                         get_last_call_transition_flag=args.get_last_call_transition_flag,
                                                         fairness_constraint = fairness_constraint,
                                                         file_root=file_root)
                    adherence_matrices[policy_option] = adherence_matrix
                    # np.save(file_root+'/adherence_log/rebuttal/adherence_'+savestring+'_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s'%(N,k,L,policy_option,args.data,args.bad_fraction, seedbase, LEARNING_MODE), adherence_matrix)
                    np.save(
                        file_root + '/logs/adherence_log/adherence_' + savestring + '_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s_bl%s_t1f%s' % (
                        N, k, L, policy_option, args.data, args.bad_fraction, seedbase, LEARNING_MODE,
                        args.buffer_length, args.get_last_call_transition_flag), adherence_matrix)
                    # adherences[policy_option].append(np.mean(np.sum(adherence_matrix, axis=1)))
                    # penalties[policy_option].append(np.mean(np.sum(penalty_matrix, axis=1)))
                    if policy_option ==66 :
                        if ep>L:
                            adherences[policy_option].append(np.mean(np.sum(adherence_matrix, axis=1)))
                            penalties[policy_option].append(np.mean(np.sum(penalty_matrix, axis=1)))
                    else:
                        adherences[policy_option].append(np.mean(np.sum(adherence_matrix, axis=1)))
                        penalties[policy_option].append(np.mean(np.sum(penalty_matrix, axis=1)))

                    # save penalty
                    # np.save(
                    #     file_root + '/logs/penalty_log/penalty_' + savestring + '_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s_bl%s_t1f%s' % (
                    #     N, k, L, policy_option, args.data, args.bad_fraction, seedbase, LEARNING_MODE,
                    #     args.buffer_length, args.get_last_call_transition_flag), penalty_matrix)



            else:
                adherence_matrix=np.zeros((N,L))
                penalty_matrix = np.zeros((N,L))
                adherence_matrices[policy_option]= adherence_matrix

                if policy_option == 6:
                    if ep > L+10:
                        adherences[policy_option] = np.mean(np.sum(adherence_matrix, axis=1))
                        penalties[policy_option] = np.mean(np.sum(penalty_matrix, axis=1))
                else:
                    adherences[policy_option] = np.mean(np.sum(adherence_matrix, axis=1))
                    penalties[policy_option] = np.mean(np.sum(penalty_matrix, axis=1))
                # adherences[policy_option]= np.mean(np.sum(adherence_matrix, axis=1))
                # penalties[policy_option] = np.mean(np.sum(penalty_matrix, axis=1))

            policy_end_time=time.time()
            policy_run_time=policy_end_time-policy_start_time
            #np.save(file_root+'/runtime/rebuttal/runtime_'+savestring+'_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s'%(N,k,L,policy_option,args.data,args.bad_fraction,seedbase, LEARNING_MODE), policy_run_time)
            np.save(file_root+'/logs/runtime/runtime_'+savestring+'_N%s_k%s_L%s_policy%s_data%s_badf%s_s%s_lr%s_bl%s_t1f%s'%(N,k,L,policy_option,args.data,args.bad_fraction,seedbase, LEARNING_MODE, args.buffer_length,args.get_last_call_transition_flag), policy_run_time)

        # write out action logs
        for policy_option in action_logs.keys():
            fname = os.path.join(args.file_root,'logs/action_logs/action_logs_'+savestring+'_N%s_k%s_L%s_data%s_badf%s_policy%s_s%s_lr%s_bl%s_t1f%s.csv'%(N,k,L, args.data,args.bad_fraction, policy_option, seedbase, LEARNING_MODE, args.buffer_length, args.get_last_call_transition_flag))
            columns = list(map(str, np.arange(N)))
            df = pd.DataFrame(action_logs[policy_option], columns=columns)
            df.to_csv(fname, index=False)

        # write out cumulative adherence logs
        for policy_option in cum_adherence.keys():
            fname = os.path.join(args.file_root,'logs/cum_adherence/cum_adherence_'+savestring+'_N%s_k%s_L%s_data%s_badf%s_policy%s_s%s_lr%s_bl%s_t1f%s.csv'%(N,k,L, args.data,args.bad_fraction, policy_option, seedbase, LEARNING_MODE, args.buffer_length, args.get_last_call_transition_flag))
            columns = list(map(str, np.arange(L)))
            df = pd.DataFrame([cum_adherence[policy_option]], columns=columns)
            df.to_csv(fname, index=False)

        # write out T matrix logs
        fname = os.path.join(args.file_root,'logs/Tmatrix_logs/Tmatrix_logs_'+savestring+'_N%s_k%s_L%s_data%s_badf%s_s%s_lr%s_bl%s_t1f%s.csv'%(N,k,L, args.data,args.bad_fraction, seedbase, LEARNING_MODE, args.buffer_length, args.get_last_call_transition_flag))
        np.save(fname, T)

    for p in range(max(policies) + 1):
        print (pname[p], ": ", np.mean(adherences[p]-np.mean(penalties[p])))
    for p in range(max(policies) + 1):
        print (pname[p], " penalty is: ", np.mean(penalties[p]))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for p in range(max(policies) + 1):
        if p == 0 or p== 1 or p==2 or p==3 or p==5 or p==19 or p==6 or p==7 or p==8:
            # print("size:", np.size(adherences[6]))
            print(pname[p], "no penalty: ", np.mean(adherences[p]),'length',len(adherences[p]))
            # print(adherences[p])
        # if p== 6 or p==7 or p==8 or p==2:
        #     print(pname[p], "no penalty: ", np.mean(adherences[p]),'length',len(adherences[p]),'std:',np.std(adherences[p],np.var(adherences)))

    end = time.time()
    print("Time taken: ", end-start)

    if args.policy < 0 and False:
        '''
        Default option (old code copy pasted under if)for running all policies code and all old stuff.
        '''
        policies_to_plot = [0,2,15,3,6,16,5,18,10,14,1]
        policies_to_plot=[0,1,2,5,6]

        bottom = 0
        labels = [pname[i] for i in policies_to_plot]
        values = [round(np.mean(np.array(adherences[i])) - bottom, 1) for i in policies_to_plot]
        errors = [np.std(np.array(adherences[i])) for i in policies_to_plot]
        # labels = ['Nobody', 'k Random', 'k Myopic', '2-day', 'Whittle', 'Yundi', 'DESPOT', 'Optimal','Oracle', 'Everybody']
        # values = [round(np.mean(adherence[0]),1), round(np.mean(adherence[2]),1), round(np.mean(adherence[3]),1), round(np.mean(adherence[6]),1),round(np.mean(adherence[5]),1), round(np.mean(adherence[10]),1), round(np.mean(adherence[9]),1), round(np.mean(adherence[4]),1),round(np.mean(adherence[8]),1),round(np.mean(adherence[1]),1)]
        # errors = [np.std(adherence0), np.std(adherence2), np.std(adherence3), np.std(adherence6),np.std(adherence5), np.std(adherence10), np.std(adherence9), np.std(adherence4),np.std(adherence8), np.std(adherence1)]

        vals = [values, errors]
        df = pd.DataFrame(vals, columns=labels)
        fname = os.path.join(args.file_root,'logs/results/results'+savestring+'_N%s_k%s_trials%s_data%s_badf%s_s%s_lr%s_bl%s_t1f%s.csv'%(N,k,N_TRIALS, args.data,args.bad_fraction, seedbase, LEARNING_MODE, args.buffer_length, args.get_last_call_transition_flag))
        df.to_csv(fname,index=False)

        '''Convert values to percentage'''
        percentages = [round(100 * (values[i] - values[0]) / (values[5] - values[0]), 0) for i in range(len(values))]
        values = percentages[1:]
        labels = labels[1:]
        errors = errors[1:]
        barPlot(labels, values, errors, ylabel='Intervention benefit as %',
                title='%s patients, %s calls per day; trials: %s ' % (N, k, N_TRIALS),
                filename=file_root + '/img/results_' + savestring + '_N%s_k%s_trials%s_data%s_s%s_lr%s.png' % (
                N, k, N_TRIALS, args.data, first_seedbase, LEARNING_MODE), root=args.file_root,
                bottom=0)



