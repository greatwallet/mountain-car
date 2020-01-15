import gym
import json
import numpy as np
import os
import os.path as osp
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime

from DDPG import Actor, Critic
from OU_Noise import OU_Noise
from Replay_Buffer import Replay_Buffer
from utils import model_deep_copy

# """ Global Parameters """
# env
env_name = "MountainCarContinuous-v0"

# GPU 
use_cuda = False
gpu_id = 1

# buffer size for replay_buffer
buffer_size = 1000000

# batch size
batch_size = 256

# seed settings
mem_seed = 17
ou_seed = 41

# learning rate
lr_critic = 2e-2
lr_actor = 3e-3

# episodes
episodes = 450

# update rate
update_every_n_steps = 20
learning_updates_per_learning_session = 10

# o-h noise parameters
mu = 0.0
theta = 0.15
sigma = 0.25

# discount 
gamma = 0.99

# gradient clipping norm
clamp_critic = 5
clamp_actor = 5

# tau for soft updating
tau_critic = 5e-3
tau_actor = 5e-3

# window size for rolling score
win = 100

# score threshold required for win
score_th = 90

# model saving path
out_path = "results"
if not osp.exists(out_path):
    os.makedirs(out_path)
    
def update_learning_rate(starting_lr, optimizer, rolling_score_list, score_th):
    """Lowers the learning rate according to how close we are to the solution"""
    if len(rolling_score_list) > 0:
        last_rolling_score = rolling_score_list[-1]
        if last_rolling_score > 0.75 * score_th:
            new_lr = starting_lr / 100.0
        elif last_rolling_score > 0.6 * score_th:
            new_lr = starting_lr / 20.0
        elif last_rolling_score > 0.5 * score_th:
            new_lr = starting_lr / 10.0
        elif last_rolling_score > 0.25 * score_th:
            new_lr = starting_lr / 2.0
        else:
            new_lr = starting_lr
        for g in optimizer.param_groups:
            g['lr'] = new_lr
    
            
def run(
    env, actor_local, actor_target, 
    critic_local, critic_target, 
    optim_actor, optim_critic, 
    memory, ou_noise, device
):
    global_step_idx = 0
    score_list = []
    rolling_score_list = []
    max_score = float('-inf')
    max_rolling_score = float('-inf')
    for i_episode in range(episodes):
        start = time.time()
        state_numpy = env.reset()
        next_state_numpy = None
        action_numpy = None
        reward = None 
        done = False
        score = 0
        
        while not done: 
            # pick an action
            state = torch.from_numpy(state_numpy).float().unsqueeze(0).to(device)
            actor_local.eval()
            with torch.no_grad():
                action_numpy = actor_local(state).cpu().data.numpy().squeeze(0)
            actor_local.train()
            # perturb as action
            action_numpy += ou_noise.sample()
            
            # conduct action
            next_state_numpy, reward, done, _ = env.step(action_numpy)
            score += reward

            # time for training and updating
            if len(memory) > batch_size and global_step_idx % update_every_n_steps == 0:
                for _ in range(learning_updates_per_learning_session):
                    # sample experience (`tensor`)
                    states_numpy, actions_numpy, rewards_numpy, next_states_numpy, dones_numpy = memory.sample()
                    states = torch.from_numpy(states_numpy).float().to(device)
                    actions = torch.from_numpy(actions_numpy).float().to(device)
                    rewards = torch.from_numpy(rewards_numpy).float().to(device)
                    next_states = torch.from_numpy(next_states_numpy).float().to(device)
                    dones = torch.from_numpy(dones_numpy).float().unsqueeze(1).to(device)
                    # 1. critic update
                    # 1.1 compute loss
                    # 1.1.1 compute target
                    with torch.no_grad():
                        next_actions = actor_target(next_states)
                        next_value = critic_target(next_states, next_actions)
                        value_target = rewards + gamma * next_value * (1.0 - dones)
                    
                    # 1.1.2. compute expected
                    value = critic_local(states, actions)
                    # 1.1.3 compute loss
                    loss_critic = F.mse_loss(value, value_target)
                    # 1.2 optimization
                    optim_critic.zero_grad()
                    loss_critic.backward()
                    
                    if clamp_critic is not None:
                        torch.nn.utils.clip_grad_norm_(
                            critic_local.parameters(), 
                            clamp_critic
                        )
                    optim_critic.step()
                    # 1.3 soft update
                    for target_param, local_param in zip(critic_target.parameters(), critic_local.parameters()):
                        target_param.data.copy_(tau_critic * local_param.data + (1.0 - tau_critic) * target_param.data)
                        
                    # 2. actor update
                    # 2.1. update learning rate
                    if done:
                        update_learning_rate(
                            starting_lr=lr_actor, 
                            optimizer=optim_actor, 
                            rolling_score_list=rolling_score_list, 
                            score_th=score_th
                        )
                    # 2.2.compute loss
                    pred_actions = actor_local(states)
                    loss_actor = -critic_local(states, pred_actions).mean()
                    # 2.3. optimization
                    optim_actor.zero_grad()
                    loss_actor.backward()
                    
                    if clamp_actor is not None:
                        torch.nn.utils.clip_grad_norm_(
                            actor_local.parameters(), 
                            clamp_actor
                        )
                    # 2.4. soft update
                    for target_param, local_param in zip(actor_target.parameters(), actor_local.parameters()):
                        target_param.data.copy_(tau_critic * local_param.data + (1.0 - tau_critic) * target_param.data)
            
            # save experience
            memory.add_experience(state_numpy, action_numpy, reward, next_state_numpy, done)
            state_numpy = next_state_numpy
            global_step_idx += 1
            
        # save and print results
        score_list.append(score)
        rolling_score = np.mean(score_list[-1 * win:])
        rolling_score_list.append(rolling_score)
        if score > max_score:
            max_score = score
        if rolling_score > max_rolling_score:
            max_rolling_score = rolling_score
        
        end = time.time()
        print("[Episode {:4d}: score: {}; rolling score: {}, max score: {}, max rolling score: {}, time cost: {:.2f}]".format(i_episode, score, rolling_score, max_score, max_rolling_score, end - start))
    
    # save results
    output = {
        "score_list": score_list, 
        "rolling_score_list": rolling_score_list, 
        "max_score": max_score, 
        "max_rolling_score": max_rolling_score
    }
    json_name = osp.join(out_path, "DDPG.json")
    with open(json_name, 'w') as f:
        json.dump(output, f, indent=4)


# """ Main Function"""
if __name__ == "__main__":
    env = gym.make(env_name)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    
    device = torch.device("cuda: %d" % gpu_id if use_cuda else "cpu")
    
    # critic
    critic_local = Critic(n_states, n_actions).to(device)
    critic_target = Critic(n_states, n_actions).to(device)
    model_deep_copy(from_model=critic_local, to_model=critic_target)
    
    optim_critic = optim.Adam(critic_local.parameters(), lr=lr_critic, eps=1e-4)
    
    memory = Replay_Buffer(buffer_size, batch_size, mem_seed)
    
    # actor
    actor_local = Actor(n_states).to(device)
    actor_target = Actor(n_states).to(device)
    model_deep_copy(from_model=actor_local, to_model=actor_target)
    
    optim_actor = optim.Adam(actor_local.parameters(), lr=lr_actor, eps=1e-4)
    
    # ou noise
    ou_noise = OU_Noise(
        size=n_actions, 
        seed=ou_seed,
        mu=mu,
        theta=theta, 
        sigma=sigma
    )
    ou_noise.reset()
    
    run(
        env=env, 
        actor_local=actor_local, 
        actor_target=actor_target, 
        critic_local=critic_local, 
        critic_target=critic_target, 
        optim_actor=optim_actor, 
        optim_critic=optim_critic, 
        memory=memory, 
        ou_noise=ou_noise, 
        device=device
    )