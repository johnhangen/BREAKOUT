#!/usr/bin/env python3

#############################################

# Base class for DQN
# This class is a base class for DQNs.

#############################################

import torch
import numpy as np
from typing import Union
from configs.config import Config
from abc import ABC, abstractmethod
from ..ExperienceReplay.UniformExperienceReplay import UniformExperienceReplay
from ..ExperienceReplay.PriorityExperienceReplay import PriorityExperienceReplay

class BaseDQN(ABC):
    """Abstract class for DQN"""

    def __init__(self, action_space:int, observation_space:int, config: Config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = action_space
        self.observation_space = observation_space

        self.policy_network = None
        self.target_network = None
        self.memory = None
        self.optimizer = None

    @abstractmethod
    def init_memory_replay(self, memory: Union[UniformExperienceReplay, PriorityExperienceReplay]) -> None:
        pass

    def init_optimizer(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.policy_network.parameters(),
            lr=self.config.Optimizer.alpha, 
        )

    def get_action(self, state:torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.epsilon:
            return torch.tensor([[np.random.randint(0, self.action_space)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                action = self.policy_network(state.to(self.device).unsqueeze(0)).max(1).indices.view(1, 1)
                #wandb.log({"Action": action})
                return action
            
    def update_epsilon(self) -> None:
        self.epsilon = max(self.config.DQN.epsilon_min, self.epsilon - self.config.DQN.epsilon_decay)

    @abstractmethod
    def init_networks(self) -> None:
        pass
            
    @abstractmethod
    def minibatch_update(self) -> None:
        pass

    @abstractmethod 
    def update_target_network(self) -> None:
        pass

    @abstractmethod
    def save_policy_network(self, path:str) -> None:
        pass

    @abstractmethod
    def load_policy_network(self, path:str) -> None:
        pass
    
    @abstractmethod
    def save_target_network(self, path:str) -> None:
        pass
    
    @abstractmethod
    def load_target_network(self, path:str) -> None:
        pass