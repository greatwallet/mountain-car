import gym
import json
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

def _init_Q_table(n_actions, init_mode):
    # action reward Table
    if init_mode == 'zeros':
        Q = defaultdict(lambda: np.zeros(n_actions))
    elif init_mode == 'random':
        Q = defaultdict(lambda: random(n_actions))
    else:
        raise ValueError("`init_mode` should be {}, {} instead of {}"
                        .format('zeros', 'random', init_mode))
        
    return Q

def _update_Q_table(Q, state, next_state, action, next_action, reward, 
                    alpha, gamma, epsilon, learning_mode, n_actions):
    if learning_mode == 'Q-learning':
        Q[state][action] = Q[state][action] + \
        alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        
    elif learning_mode == 'SARSA':
        Q[state][action] = Q[state][action] + \
        alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
        
    elif learning_mode == 'Expected-SARSA':
        # init probability for choosing action
        p_action = epsilon / n_actions \
                    * np.ones(n_actions)
        p_action[np.argmax(Q[next_state])] += 1 - epsilon
        
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
    seed, pickle_path, 
    init_mode, learning_mode, n_actions
):
    
    # set seed
    np.random.seed(seed)
    
    # env init
    env = gym.make(env_name)
    
    if n_actions is None or env_name == "MountainCar-v0":
        n_actions = 3
    
    # init Q-Table
    Q = _init_Q_table(n_actions, init_mode)
    
    score_list = []
    for episode in range(max_episodes):
        s = env.reset()
        # initialization
        s = _discretize_state(env, s, min_state_val, max_state_val)
        a = np.argmax(Q[s]) if random() > epsilon \
                    else randint(0, n_actions)
        score = 0
        done = False
        while not done:
            # Action sample
            next_s, reward, done, _ = env.step(a)
            next_s = _discretize_state(env, next_s, min_state_val, max_state_val)
            next_a = np.argmax(Q[next_s]) if random() > epsilon \
                                else randint(0, n_actions)
            
            # Update Q-Table
            Q = _update_Q_table(Q, s, next_s, a, next_a, reward, 
                    alpha, gamma, epsilon, learning_mode, n_actions)
            
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
    
    json_name = osp.join(pickle_path, '{}_{}_{}_{}_{}_{}_{}.json'
                           .format(env_name, learning_mode, init_mode,
                                  year, month, day, time))
    
    with open(json_name, 'w') as f:
        json.dump(score_list, f, indent=4)
        print("Saved score list at {}".format(json_name))
        
    return Q, score_list

def inference(
    pickle_path, env_name, epsilon, 
    min_state_val, max_state_val, 
    seed,
    save_path, 
    learning_mode='Q-learning'
):
    
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
    
#     res_path = osp.join(save_path, '{}_{}_{}'
#                            .format(env_name, learning_mode, pickle_bn))
#     env = wrappers.Monitor(env, res_path, force=True)
    
    n_actions = 3
    
    # init Q-Table
    with open(pickle_path, 'rb') as f:
        Q = pickle.load(f)
        print("Model loaded from {}".format(pickle_path))
    
    s = env.reset()
    # initialization
    s = _discretize_state(env, s, min_state_val, max_state_val)
    a = np.argmax(Q[s]) if random() > epsilon \
                else randint(0, n_actions)
    score = 0
    done = False
    step = 0
    while not done:
        # Action sample
        next_s, reward, done, _ = env.step(a)
        next_s = _discretize_state(env, next_s, min_state_val, max_state_val)
        next_a = np.argmax(Q[next_s]) if random() > epsilon \
                            else randint(0, n_actions)
        
        step += 1
        score += reward

        print("[Step: {:6d}]: action: {}, state: {}, score: {}, reward: {}"
              .format(step, a, s, score, reward))
        
        s = next_s
        a = next_a
        
    env.close()
    
    return score

# Borrowed from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#input-extraction
def get_car_location(env, screen_width):
    xmin = env.env.min_position
    xmax = env.env.max_position
    world_width = xmax - xmin
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CAR

# Borrowed from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#input-extraction
def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array')
    # Cart is in the lower half, so strip off the top and bottom of the screen
    screen_height, screen_width, _ = screen.shape
    # screen = screen[int(screen_height * 0.8), :]
    view_width = int(screen_width)
    car_location = get_car_location(env, screen_width)
    if car_location < view_width // 2:
        slice_range = slice(view_width)
    elif car_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(car_location - view_width // 2,
                            car_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, slice_range, :]
    return screen

def model_deep_copy(from_model, to_model):
    """Copies model parameters from from_model to to_model"""
    for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
        to_model.data.copy_(from_model.data.clone())