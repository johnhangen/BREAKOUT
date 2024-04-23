#!/usr/bin/env python3

#############################################

# BreakoutEnv class

# This class is used to create the Breakout environment.

#############################################

# Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from src.memory_replay import MemoryReplay

class DQN(nn.Module):
    def __init__(self, n_obs, n_action):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_action)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)

        return logits


class DQN_Network():

    def __init__(self, action_space:int, observation_space:int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyperparameters
        self.gamma: float = 0.99
        self.alpha: float = 0.0001
        self.epsilon: float = 1.0
        self.epsilon_min: float = 0.1
        self.epsilon_decay: float = 0.999
        self.batch_size: int = 32
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
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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

        Q = self.policy_network(S).gather(1, A.unsqueeze(1)).squeeze(1)
        Q_prime = self.target_network(S_prime).max(1)[0].detach()

        target = R + self.gamma * Q_prime
        loss = F.smooth_l1_loss(Q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self, t: int = 0) -> None:
        if t % self.C == 0:
            pass
            #self.target_network.load_state_dict(self.policy_network.state_dict())
            
    

    

