#!/usr/bin/env python3

#############################################

# memory replay class

# This class is used to create the memory replay used within the DQN 

#############################################

# Required Libraries
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class MemoryReplay():

    def __init__(self, max_memory:int=1000) -> None:
        self.max_memory = max_memory
        self.memory = np.empty((max_memory,), dtype=object)
        self.index = 0

    def add(self, transition:tuple) -> None:
        self.memory[self.index % self.max_memory] = Transition(
            transition[0],
            transition[1],
            transition[3],
            transition[2]
        )
        self.index += 1
    
    def sample(self, n:int=1) -> list:
        indices = np.random.randint(0, min(self.index, self.max_memory), size=n)
        return list(self.memory[indices])

    def __len__(self) -> int:
        return min(self.index, self.max_memory)
    
    def __getitem__(self, index:int) -> tuple:
        return self.memory[index % self.max_memory]
    
    def __setitem__(self, index:int, value:tuple) -> None:
        self.memory[index % self.max_memory] = value

    def __delitem__(self, index:int) -> None:
        self.memory[index % self.max_memory] = None


def main():
    pass

if __name__ == '__main__':
    main()