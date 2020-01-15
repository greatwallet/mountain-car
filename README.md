# Mountain Car
Simple Solvers for [`MountainCar-v0`](https://gym.openai.com/envs/MountainCar-v0/) and [`MountainCarContinuous-v0`](https://gym.openai.com/envs/MountainCarContinuous-v0/)  @ gym. Methods including Q-learning, SARSA, Expected-SARSA, DDPG and DQN.

## Demo
![](./figs/mountain-car.gif)

## Testing Environment
- [gym](https://gym.openai.com/)
- pytorch 1.3.1
- torchvision 0.4.2

## MountainCar-v0
Before run any script, please check out the parameters defined in the script and modify any of them as you please.
### Train with Temporal-Difference Method

```
python TD.py
```

### <b>TODO: </b>Train with DQN Method
Adapted from [REINFORCEMENT LEARNING (DQN) TUTORIAL](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) in pytorch tutorials, which originally deals with `CartPole` Problem.

DQN method has not been run and tested. 
```
python DQN.py
```

### inference with Temporal-Difference Method
```
python inference.py
```

## MountainCarContinuous-v0

### Train with DDPG
Adapted from [Deep Reinforcement Learning Algorithms with PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch) but rewritten in complete pytorch format, and redundant functions are removed.
```
python train_continuous.py
```
