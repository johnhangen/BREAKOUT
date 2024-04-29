#!/usr/bin/env python3

#############################################

# Used to run the DQN to play Breakout

#############################################

from src.DQN import DQN_Network
from src.breakout_env import BreakoutEnvAgent
from src.memory_replay import MemoryReplay

import matplotlib.pyplot as plt
import numpy as np

import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#seeding 
np.random.seed(0)
torch.manual_seed(0)

def main():
    num_episodes = 1_000
    max_memory = 1000

    # init environment
    env = BreakoutEnvAgent()
    env.init_environment()

    # init DQN
    input_shape = env.convert_observation().shape[0]
    n_actions = env.get_action_space().n
    dqn = DQN_Network(n_actions, input_shape)
    dqn.init_networks()

    # init replay memory
    memory = MemoryReplay(max_memory=max_memory)
    dqn.init_memory_replay(memory)

    for i in range(num_episodes):
        if i % 10 == 0:
            print(f"Episode: {i}, Epsilon: {round(dqn.epsilon, 4)}")

        _ = env.reset()
        S = torch.tensor(env.convert_observation(), dtype=torch.float32, device=dqn.device).unsqueeze(0)

        dqn.update_epsilon()
        
        while True:
            A = dqn.get_action(S)
            print(dqn.observation_space, dqn.action_space, A)

            S_prime, R, _, _, _ = env.step(A)
            S_prime = torch.tensor(env.convert_observation(), dtype=torch.float32, device=dqn.device).unsqueeze(0)

            R = torch.tensor(R, dtype=torch.float32, device=dqn.device)
            memory.add((S, A, R, S_prime))

            dqn.minibatch_update()

            S = S_prime

            dqn.update_target_network()

            if env.terminated or env.truncated:
                break

    env.quit()
    env.plot_rewards()

if __name__ == '__main__':
    main()