import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, n_states):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 20), 
            nn.ReLU(), 
            nn.Linear(20, 20), 
            nn.ReLU(), 
            nn.Linear(20, 20), 
            nn.ReLU(), 
            nn.Linear(20, 1)
        )
        
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states + n_actions, 20), 
            nn.ReLU(), 
            nn.Linear(20, 20), 
            nn.ReLU(), 
            nn.Linear(20, 20), 
            nn.ReLU(), 
            nn.Linear(20, n_actions)
        )
        
    def forward(self, state, action):
        return self.net(torch.cat((state, action), 1))