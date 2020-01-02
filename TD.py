import numpy as np
import os
import os.path as osp

from utils import TD_learning

# Global Variables
# learning rate
alpha = 0.1

# factor
gamma = 0.95

# maximum episodes
max_episodes = 100000

# epsilon for action choice
epsilon = 0.05

# environment: 'MountainCar-v0' or 'MountainCarContinuous-v0'
env_name = 'MountainCar-v0' 

# pickle_path
pickle_path = 'pickles'
if not osp.exists(pickle_path):
    os.makedirs(pickle_path)

# discretized state value
min_state_val = 0
max_state_val = 40

# random seed
seed = 42

# action space length
action_space_length = 3 if 'MountainCar-v0' else 1

# init mode: "zeros" or "random"
init_mode = "zeros"

# learning mode "Q-learning", "SARSA" or "Expected-SARSA"
learning_mode = "Q-learning"

if __name__ == "__main__":
    TD_learning(
        env_name=env_name,
        alpha=alpha, 
        gamma=gamma, 
        epsilon=epsilon, 
        max_episodes=max_episodes, 
        min_state_val=min_state_val, 
        max_state_val=max_state_val, 
        action_space_length=3, 
        seed=seed, 
        pickle_path=pickle_path, 
        init_mode=init_mode, 
        learning_mode=learning_mode
    )
    