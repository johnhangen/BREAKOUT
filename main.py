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

print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def main():
    num_episodes = 10_000
    max_memory = 10_000

    config = Config.load_config("configs/starting_params.yaml")

    # init environment
    env = BreakoutEnvAgent(
        #render_mode="human"
    )
    env.init_environment()

    # init DQN
    input_shape = env.convert_observation().shape[0]
    n_actions = env.get_action_space().n

    print(n_actions)

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
            "Device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        },
        #mode="disabled",
    )

    dqn = DQN_Network(n_actions, input_shape, config)
    dqn.init_networks()

    memory = MemoryReplay(max_memory=max_memory)
    dqn.init_memory_replay(memory)

    for i in range(num_episodes):
        if i % 10 == 0 and i != 0:
            print(f"Episode: {i}, Epsilon: {round(dqn.epsilon, 4)}")
            dqn.save_policy_network("model/DQN_policy.pt")
            dqn.save_target_network("model/DQN_target.pt")

        _ = env.reset()
        S = env.convert_observation()

        running_rewards = 0 
        
        while True:
            A = dqn.get_action(S)

            S_prime, R, _, _, _ = env.step(A)
            S_prime = env.convert_observation()

            wandb.log({"reward": R})
            running_rewards += R
            R = max(-1, min(1, R)) 
            
            R = torch.tensor([[R]], dtype=torch.float32, device=dqn.device)
            memory.add((S, A, R, S_prime))

            dqn.minibatch_update()

            dqn.update_target_network()

            dqn.update_epsilon()

            S = S_prime

            if env.terminated or env.truncated:
                break

        wandb.log({"episode_reward": running_rewards, "episode": i})

    # ending process
    env.quit()
    #env.plot_rewards()
    dqn.save_policy_network("model/DQN_policy.pt")
    dqn.save_target_network("model/DQN_target.pt")

if __name__ == '__main__':
    main()