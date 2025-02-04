#!/usr/bin/env python3

#############################################

# ExperienceReplay class
# This class is a base class for experience replays.

#############################################

from abc import ABC, abstractmethod
from collections import namedtuple
from configs.config import Config
import random

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))


class ExperienceReplay(ABC):
    """Abstract class for experience replays"""

    def __init__(self, config: Config):
        self.config = config
        self.seed = self.config.MemoryReplay.seed
        self.max_memory = self.config.MemoryReplay.max_memory

        self._size = 0

    def size(self) -> int:
        return self._size
    
    @property
    def seed(self) -> int:
        return self._seed
    
    @seed.setter
    def seed(self, seed:int) -> None:
        self._seed = seed
        random.seed(seed)
    
    @abstractmethod
    def sample(self, batch_size: int = 32) -> list[Transition]:
        pass

    @abstractmethod
    def add(self, *args) -> None:
        pass
