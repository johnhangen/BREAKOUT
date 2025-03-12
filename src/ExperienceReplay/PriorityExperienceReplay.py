#!/usr/bin/env python3

#############################################

# priority memory replay class
# This class is used to create the priority memory replay used within the DQN 

#############################################

# Required Libraries
from collections import deque
from typing import List, Tuple
import random
import numpy as np

from configs import config
from .ExperienceReplay import ExperienceReplay, Transition

class PriorityExperienceReplay(ExperienceReplay):

    def __init__(self, config: config) -> None:
        super().__init__(config=config)
        self.memory = deque(maxlen=self.max_memory)
        self.priorities = deque(maxlen=self.max_memory)

    def add(self, *args) -> None:
        self.memory.append(Transition(*args))
        max_priority = max(self.priorities, default=1.0)
        self.priorities.append(max_priority)
    
    def sample(self, batch_size: int = 32) -> list[Transition]:
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def sample(self, batch_size: int = 32) -> Tuple[List[Transition], np.ndarray]:
        if len(self.memory) == 0:
            return [], np.array([]), np.array([])

        priorities = np.array(self.priorities, dtype=np.float32) ** self.config.MemoryReplay.alpha
        probs = priorities / np.sum(priorities)

        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
        transitions = [self.memory[i] for i in indices]
        importance_weights = (len(self.memory) * probs[indices]) ** (-self.config.MemoryReplay.beta)
        importance_weights /= importance_weights.max()

        return transitions, indices
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = abs(td_error)

    def __len__(self) -> int:
        return len(self.memory)
    
    def __getitem__(self, index:int) -> Transition:
        return self.memory[index]
    
    def __setitem__(self, index:int, value:Transition) -> None:
        self.memory[index] = value

    def __delitem__(self, index:int) -> None:
        self.memory[index] = None
        self.size -= 1