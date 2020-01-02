import gym
import numpy as np
import os.path as osp
import pickle
import time

from collections import defaultdict
from datetime import datetime
from gym import wrappers
from numpy.random import random, randint

def _discretize_state(env, state, min_val, max_val):
    """ discretize the continuous state to [min_val, max_val]
    
    """
    if not isinstance(state, np.ndarray):
        raise ValueError("Expecting state to be {}, but received {}"
                        .format(np.ndarray, type(state)))
    
    if len(state.squeeze().shape) > 2:
        raise ValueError("Expecting shape of state to be {} or {}, \
                         but received shape of {}"
                        .format(1, 2, state.shape))
    
    low = env.observation_space.low
    high = env.observation_space.high
    
    min_val = np.array(min_val)
    max_val = np.array(max_val)
    
    ret = np.round(max_val * (state - low) / (high - low) + min_val)
    
    return tuple(ret.astype('int'))

def _init_Q_table(env, action_space_length, init_mode):
    low = env.observation_space.low
    high = env.observation_space.high
    # action reward Table
    if init_mode == 'zeros':
        Q = defaultdict(lambda: np.zeros(action_space_length))
    elif init_mode == 'random':
        Q = defaultdict(lambda: random(action_space_length) * (high - low) + low)
    else:
        raise ValueError("`init_mode` should be {}, {} instead of {}"
                        .format('zeros', 'random', init_mode))
        
    return Q

def _update_Q_table(Q, state, next_state, action, next_action, reward, 
                    alpha, gamma, learning_mode):
    if learning_mode == 'Q-learning':
        Q[state][action] = Q[state][action] + \
        alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        
    elif learning_mode == 'SARSA':
        Q[state][action] = Q[state][action] + \
        alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
        
    elif learning_mode == 'Expected-SARSA':
        # init probability for choosing action
        p_action = epsilon / action_space_length \
                    * np.ones(action_space_length)
        p_action[np.max(Q[next_state])] += 1 - epsilon
        
        # calculate expected action return
        expected_action_return = np.average(Q[next_state], weights=p_action)
        
        Q[state][action] = Q[state][action] + \
        alpha * (reward + gamma * expected_action_return - Q[state][action])  
        
    else:
        raise ValueError("`learning_mode` should be {}, {} or {} instead of {}"
                        .format('Q-learning', 'SARSA', 'Expected-SARSA', learning_mode))
    return Q
        
def TD_learning(
    env_name, alpha, gamma, epsilon, 
    max_episodes, min_state_val, max_state_val, 
    action_space_length, seed, pickle_path, 
    init_mode, learning_mode='Q-learning'):
    
    # set seed
    np.random.seed(seed)
    
    # env init
    env = gym.make(env_name)
    
    # init Q-Table
    Q = _init_Q_table(env, action_space_length, init_mode)
    
    score_list = []
    for episode in range(max_episodes):
        s = env.reset()
        # initialization
        s = _discretize_state(env, s, min_state_val, max_state_val)
        a = np.argmax(Q[s]) if random() > epsilon \
                    else randint(0, action_space_length)
        score = 0
        done = False
        while not done:
            # Action sample
            next_s, reward, done, _ = env.step(a)
            next_s = _discretize_state(env, next_s, min_state_val, max_state_val)
            next_a = np.argmax(Q[next_s]) if random() > epsilon \
                                else randint(0, action_space_length)
            
            # Update Q-Table
            Q = _update_Q_table(Q, s, next_s, a, next_a, reward, 
                    alpha, gamma, learning_mode)
            
            score += reward
            s = next_s
            a = next_a
            
        score_list.append(score)
        print('[Episode {:06d}] score: {}, best_score: {}'
             .format(episode, score, max(score_list)))
              
    env.close()
    
    # save pickle
    now = datetime.now() # current date and time
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    time = now.strftime("%H_%M_%S")
    pickle_name = osp.join(pickle_path, '{}_{}_{}_{}_{}_{}_{}.pickle'
                           .format(env_name, learning_mode, init_mode,
                                  year, month, day, time))
    with open(pickle_name, 'wb') as f:
        pickle.dump(dict(Q), f)
        print("Saved model at {}".format(pickle_name))
    
    return Q, score_list

def inference(
    pickle_path, env_name, epsilon, 
    min_state_val, max_state_val, 
    action_space_length, seed,
    save_path, 
    learning_mode='Q-learning'):
    
    # set seed
    np.random.seed(seed)
    
    # env init
    env = gym.make(env_name)
    # save result path
    pickle_bn = osp.basename(pickle_path)
    pickle_bn, _ = osp.splitext(pickle_bn)
    
    now = datetime.now() # current date and time
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    time = now.strftime("%H_%M_%S")
    
    res_path = osp.join(save_path, '{}_{}_{}'
                           .format(env_name, learning_mode, pickle_bn))
    env = wrappers.Monitor(env, res_path, force=True)
    
    # init Q-Table
    with open(pickle_path, 'rb') as f:
        Q = pickle.load(f)
        print("Model loaded from {}".format(pickle_path))
    
    s = env.reset()
    # initialization
    s = _discretize_state(env, s, min_state_val, max_state_val)
    a = np.argmax(Q[s]) if random() > epsilon \
                else randint(0, action_space_length)
    score = 0
    done = False
    step = 0
    while not done:
        # Action sample
        next_s, reward, done, _ = env.step(a)
        next_s = _discretize_state(env, next_s, min_state_val, max_state_val)
        next_a = np.argmax(Q[next_s]) if random() > epsilon \
                            else randint(0, action_space_length)
        
        step += 1
        score += reward

        print("[Step: {:6d}]: action: {}, state: {}, score: {}, reward: {}"
              .format(step, a, s, score, reward))
        
        s = next_s
        a = next_a
        
    env.close()
    

    
    return score