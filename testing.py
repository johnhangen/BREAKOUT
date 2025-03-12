#!/usr/bin/env python3

#############################################

# Used to run the DQN to play Breakout

#############################################

from src.Model.DQN import DQN_Network
from src.breakout_env import BreakoutEnvAgent
from src.ExperienceReplay.UniformExperienceReplay import UniformExperienceReplay
from src.ExperienceReplay.PriorityExperienceReplay import PriorityExperienceReplay

from configs.config import Config
import wandb

import numpy as np

import torch
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
        config=config
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
    memory = UniformExperienceReplay(config=config)
    dqn.init_memory_replay(memory)

    dqn.load_policy_network("model\DQN_policy (1).pt")
    dqn.load_target_network("model\DQN_target (1).pt")

    _ = env.reset()
    S = env.convert_observation()

    for i in range(num_episodes):
        if i % 10 == 0 and i != 0:
            print(f"Episode: {i}, Epsilon: {round(dqn.epsilon, 4)}")

        running_rewards = 0
        
        cnt = 0
        while True:
            cnt += 1
            A = dqn.get_action(S)

            if cnt < 60:
                file_path = f'figs/breakout_screenshot_{cnt}.png'
                env.screenshot(file_path=file_path)

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