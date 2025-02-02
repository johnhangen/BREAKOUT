#!/usr/bin/env python3

#############################################

# Used to run the DQN to play Breakout

#############################################

from src.DQN import DQN_Network
from src.breakout_env import BreakoutEnvAgent
from src.memory_replay import MemoryReplay

from configs.config import Config
import wandb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch

import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# seeding 
np.random.seed(0)
torch.manual_seed(0)

def main():
    num_episodes = 1_0
    max_memory = 100

    config = Config.load_config(r"configs\testing_params.yaml")

    # init environment
    env = BreakoutEnvAgent(
        render_mode="human"
    )
    env.init_environment()

    # init DQN
    input_shape = env.convert_observation().shape[0]
    n_actions = env.get_action_space().n

    wandb.init(
        mode="disabled"
    )

    dqn = DQN_Network(n_actions, input_shape, config)
    dqn.init_networks()

    # init replay memory
    memory = MemoryReplay(max_memory=max_memory)
    dqn.init_memory_replay(memory)

    dqn.load_policy_network("model\DQN_policy.pt")
    dqn.load_target_network("model\DQN_target.pt")

    _ = env.reset()
    S = env.convert_observation()

    for i in range(num_episodes):
        if i % 10 == 0 and i != 0:
            print(f"Episode: {i}, Epsilon: {round(dqn.epsilon, 4)}")

        running_rewards = 0
        
        while True:
            A = dqn.get_action(S)

            S_prime, R, _, _, _ = env.step(A)
            S_prime = env.convert_observation()

            S = S_prime

            running_rewards += R

            if env.terminated or env.truncated:
                print(running_rewards)
                running_rewards = 0
                _ = env.reset()
                S = env.convert_observation()

    # ending process
    env.quit()

if __name__ == '__main__':
    main()