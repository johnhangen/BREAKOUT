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

# TODO: Implement the CNN from the paper
class DQN(nn.Module):
    def __init__(self, n_obs, n_action):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(n_obs, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, n_action)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
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
        self.policy_network = DQN(self.observation_space, self.action_space).to(self.device)
        self.target_network = DQN(self.observation_space, self.action_space).to(self.device)
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

        batch = self.memory.sample(self.batch_size)
        S, A, R, S_prime = zip(*batch)

        S = torch.cat(S).to(self.device)
        A = torch.tensor(A, dtype=torch.int64).to(self.device)
        R = torch.tensor(R).to(self.device)
        S_prime = torch.cat(S_prime).to(self.device)

        Q = self.policy_network(S).gather(1, A.unsqueeze(1))
        Q_prime = self.target_network(S_prime).max(1)[0].detach()

        target = R + self.gamma * Q_prime
        loss = F.smooth_l1_loss(Q, target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 1)
        self.optimizer.step()

    def update_target_network(self, t: int = 0) -> None:
        if t % self.C == 0:
            pass
            #self.target_network.load_state_dict(self.policy_network.state_dict())

    #TODO: I would like a method so the user can see a frame of the game

    def save_policy_network(self, path:str) -> None:
        torch.save(self.policy_network.state_dict(), path)

    def load_policy_network(self, path:str) -> None:
        self.policy_network.load_state_dict(torch.load(path))
    
    def save_target_network(self, path:str) -> None:
        torch.save(self.target_network.state_dict(), path)
    
    def load_target_network(self, path:str) -> None:
        self.target_network.load_state_dict(torch.load(path))
            
    

    

