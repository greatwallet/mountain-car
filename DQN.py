import gym
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from itertools import count
from PIL import Image

from dqn_models import DQN, ReplayMemory, Transition
from utils import get_screen

# environment: 'MountainCar-v0' or 'MountainCarContinuous-v0'
env_name = 'MountainCar-v0' 
device = torch.device("cpu")
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
num_episoeds = 50

resize = T.Compose([
    T.ToTensor(),
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.CUBIC),
    T.ToTensor()
])

if __name__ == "__main__":
    env = gym.make(env_name)
    init_screen = get_screen(env)
    screen_height, screen_width, _ = init_screen.shape
    
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)
    
    steps_done = 0
    
    episode_durations = []
    
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = resize(get_screen(env)).unsqueeze(0).to(device)
        current_screen = resize(get_screen(env)).unsqueeze(0).to(device)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            # action = select_action(state)
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad:
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    action = policy_net(state).max(1)[1].view(1, 1).item()
            else:
                action = random.randrange(n_actions)
            
            _, reward, done, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            
            # Observe new state
            last_scrren = current_screen
            current_screen = resize(get_screen(env)).unsqueeze(0).to(device)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None
            
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            
            # Move to the next state
            state = next_state
            
            # Perform one step of optimization (on the target network)
            if len(memory) < BATCH_SIZE:
                pass
            else:
                transitions = memory.sample(BATCH_SIZE)
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))
                
                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = policy_net(state_batch).gather(1, action_batch)
                
                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1)[0].
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch
                
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                
            if done:
                break            
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())