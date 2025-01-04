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

import numpy as np
import torch.utils

from src.memory_replay import MemoryReplay

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        dummy_input = torch.zeros(1, 3, 84, 84)
        conv_output_size = self.features(dummy_input).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

class DQN_Network():

    def __init__(self, action_space:int, observation_space:int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyperparameters
        self.gamma: float = 0.99
        self.alpha: float = 0.0001
        self.epsilon: float = 1.0
        self.epsilon_min: float = 0.1
        self.epsilon_decay: float = 0.001
        self.batch_size: int = 500
        self.memory_size: int = 100_000
        self.C:int = 1000

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

    def init_memory_replay(self, memory) -> None:
        self.memory = memory

    def init_optimizer(self) -> None:
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.alpha)

    def init_networks(self) -> None:
        self.policy_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.init_optimizer()

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def get_action(self, state:torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.epsilon:
            return torch.tensor([[np.random.randint(0, self.action_space)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_network(state).max(1).view(1, 1)
            
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
        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch        

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()      

    def update_target_network(self, t: int = 0) -> None:
        if t % self.C == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_policy_network(self, path:str) -> None:
        torch.save(self.policy_network.state_dict(), path)

    def load_policy_network(self, path:str) -> None:
        self.policy_network.load_state_dict(torch.load(path))
    
    def save_target_network(self, path:str) -> None:
        torch.save(self.target_network.state_dict(), path)
    
    def load_target_network(self, path:str) -> None:
        self.target_network.load_state_dict(torch.load(path))
            
    

    

