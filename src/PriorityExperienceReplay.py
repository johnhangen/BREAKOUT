#!/usr/bin/env python3

#############################################

# priority memory replay class
# This class is used to create the priority memory replay used within the DQN 

#############################################

# Required Libraries
from collections import deque
import random

from configs import config
from .ExperienceReplay import ExperienceReplay, Transition

class PriorityExperienceReplay(ExperienceReplay):

    def __init__(self, config: config) -> None:
        super().__init__(config=config)
        self.memory = deque(maxlen=self.max_memory)

    def add(self, *args) -> None:
        self.memory.append(Transition(*args))
        self.size += 1
    
    def sample(self, batch_size: int = 32) -> list[Transition]:
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self) -> int:
        return len(self.memory)
    
    def __getitem__(self, index:int) -> Transition:
        return self.memory[index]
    
    def __setitem__(self, index:int, value:Transition) -> None:
        self.memory[index] = value

    def __delitem__(self, index:int) -> None:
        self.memory[index] = None
        self.size -= 1