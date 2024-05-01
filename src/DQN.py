#!/usr/bin/env python3

#############################################

# DQN class

# This class is used to create the DQN.

#############################################

# Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import torch.utils

from src.memory_replay import MemoryReplay

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(49, 49)  
        self.fc2 = nn.Linear(49, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
        self.batch_size: int = 1000
        self.memory_size: int = 10_000
        self.C:int = 500

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

    def get_action(self, state:torch.Tensor) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_space)
        else:
            with torch.no_grad():
                return torch.argmax(self.policy_network(state)).item()
            
    def minibatch_update(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        # Sample random minibatch of transitions from D
        batches = self.memory.sample(self.batch_size)
        
        #S, A, R, S_prime = zip(*batch)

        #S = torch.cat(S).to(self.device)
        #A = torch.tensor(A, dtype=torch.int64).to(self.device)
        #R = torch.tensor(R).to(self.device)
        #S_prime = torch.cat(S_prime).to(self.device)

        # TODO: make batch trainning work
        for _, batch in enumerate(batches):
            S, A, R, S_prime = batch

            # compute Q(s_t, a)
            Q = self.policy_network(S)[:, A]
            Q_prime = self.target_network(S_prime).max(1)[0].detach()

            target = R + self.gamma * Q_prime
            loss = F.smooth_l1_loss(Q, target)

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
            
    

    

