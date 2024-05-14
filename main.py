#!/usr/bin/env python3

#############################################

# Used to run the DQN to play Breakout

#############################################

from src.DQN import DQN_Network
from src.breakout_env import BreakoutEnvAgent
from src.memory_replay import MemoryReplay

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
    num_episodes = 1000
    max_memory = 10_000

    # init database
    df = pd.DataFrame(
        columns=[
        "Episode",
        "Epsilon Value",
        "gamma",
        "alpha",
        "epsilon",
        "epsilon_min",
        "epsilon_decay",
        "batch_size",
        "C",
        "Memory Size",
        "Rewards",
        "Wall Time"
    ])

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

    start = time.time()

    for i in range(num_episodes):
        if i % 10 == 0 and i != 0:
            print(f"Episode: {i}, Epsilon: {round(dqn.epsilon, 4)}")

        _ = env.reset()
        S = torch.tensor(env.convert_observation(), dtype=torch.float32, device=dqn.device).unsqueeze(0)

        dqn.update_epsilon()

        running_rewards = 0
        
        while True:
            A = dqn.get_action(S)

            S_prime, R, _, _, _ = env.step(A)
            S_prime = torch.tensor(env.convert_observation(), dtype=torch.float32, device=dqn.device).unsqueeze(0)

            running_rewards += R

            R = torch.tensor(R, dtype=torch.float32, device=dqn.device)
            assert A <= n_actions
            memory.add((S, A, R, S_prime))

            dqn.minibatch_update()

            S = S_prime

            dqn.update_target_network()

            if env.terminated or env.truncated:
                break

        df = df._append({
            "Episode": i,
            "Epsilon Value": dqn.epsilon,
            "gamma": dqn.gamma,
            "alpha": dqn.alpha,
            "epsilon": dqn.epsilon,
            "epsilon_min": dqn.epsilon_min,
            "epsilon_decay": dqn.epsilon_decay,
            "batch_size": dqn.batch_size,
            "C": dqn.C,
            "Memory Size": dqn.memory_size,
            "Rewards": running_rewards,
            "Wall Time": start - time.time() }, ignore_index=True)
        
        dqn.save_policy_network('model/DQN_policy.pt')
        dqn.save_target_network('model/DQN_target.pt')
        df.to_csv("data/DQN_Breakout.csv")

    # ending process
    env.quit()
    env.plot_rewards()

if __name__ == '__main__':
    main()