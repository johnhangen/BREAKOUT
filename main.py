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
    num_episodes = 1_00
    max_memory = 10_000

    config = Config.load_config("/content/BREAKOUT/configs/starting_params.yaml")

    # init environment
    env = BreakoutEnvAgent(
        #render_mode="human"
    )
    env.init_environment()

    # init DQN
    input_shape = env.convert_observation().shape[0]
    n_actions = env.get_action_space().n

    wandb.init(
        config={
            "gamma": config.DQN.gamma,
            "alpha": config.DQN.alpha,
            "epsilon": config.DQN.epsilon,
            "epsilon_min": config.DQN.epsilon_min,
            "epsilon_decay": config.DQN.epsilon_decay,
            "batch_size": config.DQN.batch_size,
            "memory_size": config.DQN.memory_size,
            "C": config.DQN.C,
            "num_episodes": num_episodes,
            "max_memory": max_memory,
            "n_actions": n_actions,
            "input_shape": input_shape,
        },
        #mode="disabled",
    )

    dqn = DQN_Network(n_actions, input_shape, config)
    dqn.init_networks()

    # init replay memory
    memory = MemoryReplay(max_memory=max_memory)
    dqn.init_memory_replay(memory)

    for i in range(num_episodes):
        if i % 10 == 0 and i != 0:
            print(f"Episode: {i}, Epsilon: {round(dqn.epsilon, 4)}")

        _ = env.reset()
        S = env.convert_observation()

        dqn.update_epsilon()

        running_rewards = 0
        
        while True:
            A = dqn.get_action(S)

            S_prime, R, _, _, _ = env.step(A)
            S_prime = env.convert_observation()
            wandb.log(
                {"reward": R}
            )

            running_rewards += R

            R = torch.tensor([[R]], dtype=torch.float32, device=dqn.device)
            assert A <= n_actions
            memory.add((S, A, R, S_prime))

            dqn.minibatch_update()

            S = S_prime

            dqn.update_target_network()
            dqn.update_epsilon()

            if env.terminated or env.truncated:
                break

    # ending process
    env.quit()
    env.plot_rewards()
    dqn.save_policy_network("model\DQN_policy.pt")
    dqn.save_target_network("model\DQN_target.pt")

if __name__ == '__main__':
    main()