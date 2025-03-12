#!/usr/bin/env python3

#############################################

# DDQN class

# This class is used to create the DDQN.

#############################################

# Required Libraries
import torch
import torch.nn as nn
from . import DQN
import wandb

import numpy as np
import torch.utils
from typing import Union

from ..ExperienceReplay.UniformExperienceReplay import UniformExperienceReplay
from ..ExperienceReplay.PriorityExperienceReplay import PriorityExperienceReplay
from ..ExperienceReplay.Transition import Transition
from .Base_DQN import BaseDQN
from .Model import DQN
from configs.config import Config

class DQN_Network(BaseDQN):

    def __init__(self, action_space:int, observation_space:int, config:Config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.count = 0
        self.action_space: int = action_space
        self.observation_space: int = observation_space
        self.policy_network: DQN = None
        self.target_network: DQN = None
        self.memory = None
        self.optimizer = None

    def init_memory_replay(self, memory: Union[UniformExperienceReplay, PriorityExperienceReplay]) -> None:
        self.memory = memory

    def init_networks(self) -> None:
        self.policy_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.policy_network.train()
        self.target_network.train()
        wandb.watch(self.policy_network, log_freq=1000)

        self.init_optimizer()
            
    def minibatch_update(self) -> None:
        self.count += 1
        if len(self.memory) < self.config.DQN.batch_size:
            return
        
        transitions = self.memory.sample(self.config.DQN.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None]).to(self.device)

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.tensor(batch.reward, device=self.device).float()
        
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.config.DQN.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.config.DQN.gamma) + reward_batch    

        td_error = (expected_state_action_values - state_action_values).detach()  

        criterion = nn.HuberLoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), self.config.DQN.grad_clip_val)
        self.optimizer.step()    

        if self.count % self.config.ENV.log_freq == 0:
            wandb.log({"loss": loss,
                    "Epsilon":self.epsilon,
                    "td_error_mean": td_error.mean().item(),
                    "td_error_max": td_error.max().item(),
                    "td_error_min": td_error.min().item(),
                    "q_value_mean": state_action_values.mean().item(),
                    "q_value_max": state_action_values.max().item(),
                    "q_value_min": state_action_values.min().item()
                    })

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
