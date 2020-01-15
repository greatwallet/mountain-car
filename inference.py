import numpy as np
import os
import os.path as osp

from utils import inference

# Global Variables
# epsilon for action choice
epsilon = 0.05

# environment: 'MountainCar-v0' or 'MountainCarContinuous-v0'
env_name = 'MountainCar-v0' 

# pickle_path
pickle_path = osp.join('pickles', 'latest.pickle')

# discretized state value
min_state_val = 0
max_state_val = 40

# random seed
seed = 42


# learning mode "Q-learning", "SARSA" or "Expected-SARSA"
learning_mode = "Q-learning"

# save path
save_path = None # 'results'
if save_path is not None and not osp.exists(save_path):
    os.makedirs(save_path)
    
if __name__ == "__main__":
    score = inference(
        pickle_path=pickle_path, 
        env_name=env_name, 
        epsilon=epsilon, 
        min_state_val=min_state_val, 
        max_state_val=max_state_val,
        seed=seed, 
        save_path=save_path, 
        learning_mode=learning_mode
    )
    print("Score: {} (Model: {}; Env: {}) ".format(score, pickle_path, env_name))
    