#!/usr/bin/env python3

#############################################

# DQN class

# This class is used to create the DQN.

#############################################

# Required Libraries
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

import numpy as np
import torch.utils
from typing import Union

from .UniformExperienceReplay import UniformExperienceReplay
from .PriorityExperienceReplay import PriorityExperienceReplay
from configs.config import Config

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, input_channels=4, num_actions=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #print(f"Shape before flattening: {x.shape}")
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQN_Network():

    def __init__(self, action_space:int, observation_space:int, config:Config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyperparameters
        self.gamma: float = self.config.DQN.gamma
        self.epsilon: float = self.config.DQN.epsilon
        self.epsilon_min: float = self.config.DQN.epsilon_min
        self.epsilon_decay: float = self.config.DQN.epsilon_decay
        self.batch_size: int = self.config.DQN.batch_size
        self.C:int = self.config.DQN.C

        #enviroment vars
        self.action_space: int = action_space
        self.observation_space: int = observation_space

        # networks
        self.policy_network: DQN = None
        self.target_network: DQN = None

        # memory replay
        self.memory = None

        # optimizer
        self.optimizer = None

    def init_memory_replay(self, memory: Union[UniformExperienceReplay, PriorityExperienceReplay]) -> None:
        self.memory = memory

    def init_optimizer(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.policy_network.parameters(),
            lr=self.config.Optimizer.alpha, 
            #alpha=self.config.Optimizer.squared_gradient_momentum,
            #eps=self.config.Optimizer.min_squared_gradient,
            #momentum=self.config.Optimizer.gradient_momentum,
            #centered=False
        )
    def init_networks(self) -> None:
        self.policy_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.policy_network.train()
        self.target_network.train()
        wandb.watch(self.policy_network, log_freq=100)

        self.init_optimizer()

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def get_action(self, state:torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.epsilon:
            return torch.tensor([[np.random.randint(0, self.action_space)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                action = self.policy_network(state.to(self.device).unsqueeze(0)).max(1).indices.view(1, 1)
                wandb.log({"Action": action})
                return action
            
    def minibatch_update(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None]).to(self.device)

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.tensor(batch.reward, device=self.device).float()
        
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        wandb.log({
            "q_value_mean": state_action_values.mean().item(),
            "q_value_max": state_action_values.max().item(),
            "q_value_min": state_action_values.min().item(),
        })
        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch    

        td_error = (expected_state_action_values - state_action_values).detach()
        wandb.log({
            "td_error_mean": td_error.mean().item(),
            "td_error_max": td_error.max().item(),
            "td_error_min": td_error.min().item(),
        })    

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), self.config.DQN.grad_clip_val)
        self.optimizer.step()    

        wandb.log({"loss": loss,
                   "Epsilon":self.epsilon})

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_policy_network(self, path:str) -> None:
        torch.save(self.policy_network.state_dict(), path)

    def load_policy_network(self, path:str) -> None:
        self.policy_network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    def save_target_network(self, path:str) -> None:
        torch.save(self.target_network.state_dict(), path)
    
    def load_target_network(self, path:str) -> None:
        self.target_network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
