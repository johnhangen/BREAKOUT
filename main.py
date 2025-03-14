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
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# seeding 
np.random.seed(0)
torch.manual_seed(0)

def main():
    print("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load_config("configs/starting_params.yaml")

    # init environment
    env = BreakoutEnvAgent(config=config)
    env.init_environment()

    # init DQN
    input_shape = env.convert_observation().shape[0]
    n_actions = env.get_action_space().n
    #env.get_action_mappings()

    # TODO: this needs to be updated to reflect all params and if we are using wandb
    wandb.init(
        config={
            "gamma": config.DQN.gamma,
            "alpha": config.Optimizer.alpha,
            "epsilon": config.DQN.epsilon,
            "epsilon_min": config.DQN.epsilon_min,
            "epsilon_decay": config.DQN.epsilon_decay,
            "batch_size": config.DQN.batch_size,
            "C": config.DQN.C,
            "num_episodes": config.DQN.num_episodes,
            "max_memory": config.MemoryReplay.max_memory,
            "n_actions": n_actions,
            "input_shape": input_shape,
            "Device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        },
        mode="disabled",
    )

    dqn = DQN_Network(n_actions, input_shape, config)
    dqn.init_networks()

    # init experience replay
    if config.MemoryReplay.priority_replay:
        memory = PriorityExperienceReplay(config=config)
    else:
        memory = UniformExperienceReplay(config=config)
    dqn.init_memory_replay(memory)

    highest_reward = 0.0
    since = time.time()
    c = 0

    for i in range(config.DQN.num_episodes):
        _ = env.reset()
        S = env.convert_observation()

        running_rewards = 0.0
        
        cnt = 0
        while True:
            cnt += 1
            c += 1
            A = dqn.get_action(S)

            S_prime, R, _, _, _ = env.step(A)
            S_prime = env.convert_observation()

            running_rewards += R

            memory.add(S, A, S_prime, torch.tensor([[R]], dtype=torch.float32, device=dqn.device))

            dqn.minibatch_update()
            dqn.update_epsilon()
            S = S_prime

            if c % config.DQN.C == 0:
                dqn.update_target_network()

            if env.terminated or env.truncated:
                break

        wandb.log({
            "episode_reward": running_rewards, 
            "episode": i
            #"memory_len":len(memory)
            })

        if i % 100 == 0 and i != 0:
            print(f"Episode: {i}, Epsilon: {round(dqn.epsilon, 4)}, HIGH RW: {highest_reward}")
            if running_rewards > highest_reward:
                highest_reward = running_rewards
                dqn.save_policy_network("model/DQN_policy.pt")
                dqn.save_target_network("model/DQN_target.pt")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {highest_reward:.4f}')

    # ending process
    env.quit()
    #env.plot_rewards()
    dqn.save_policy_network("model/DQN_policy.pt")
    dqn.save_target_network("model/DQN_target.pt")

if __name__ == '__main__':
    main()