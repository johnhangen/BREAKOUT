#!/usr/bin/env python3

#############################################

# BreakoutEnv class

# This class is used to create the Breakout environment.

#############################################

# Required Libraries
import gym
from gym.utils.play import play
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from skimage.transform import resize
import torchvision.transforms as T
import torch
import wandb

from configs.config import Config

# housekeeping
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plt.style.use('ggplot')

class BreakoutEnvAgent():
    
    def __init__(self, config: Config) -> None:
        # Config
        self.config = config

        # MDP environment
        self.observation = None
        self.info = None
        self.reward = None
        self.terminated: bool = None
        self.truncated: bool = None

        # make the environment
        self._seed: int = self.config.ENV.seed
        self.render_mode: bool = self.config.ENV.render_mode

        # image inits
        self.observation_queue = torch.zeros(4, 84, 84)
        self.previous_frame = torch.zeros(1, 84, 84)
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((84, 84)),
            T.Normalize(0.5, 0.5),
            T.ConvertImageDtype(torch.float32),
        ])

        # graphing vars
        self.plot_rewards_bool = self.config.ENV.plot_rewards_bool
        self.reward_running: float = 0.0
        self.rewards: np.array = np.array([])
        

    def init_environment(self) -> None:
        self.env = gym.make(
                            'ALE/Breakout-v5', 
                            render_mode=self.render_mode
                            )
        self.env.seed(self.seed)
        self.observation, self.info = self.env.reset(seed=self.seed)

    def wandb_image(self):
        wandb.log({"Examples": wandb.Image(self.observation)})

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, seed:int) -> None:
        '''NO LONGER SUPPORTED'''
        self.env.seed(seed)

    def reset(self) -> tuple:
        if self.plot_rewards_bool:
            self.rewards = np.append(self.rewards, self.reward_running)
            self.reward_running = 0.0
        self.terminated = False
        self.truncated = False
        self.observation, self.info = self.env.reset(seed=self._seed)
        return self.observation, self.info

    def get_action_space(self) -> gym.spaces.Discrete:
        return self.env.action_space
    
    def get_action_mappings(self) -> None:
        '''Assert actions needed'''
        print(self.env.unwrapped.get_action_meanings())

    def get_observation_space(self) -> gym.spaces.Box:
        return self.env.observation_space
    
    def get_reward_range(self) -> tuple:
        return self.env.reward_range
    
    def convert_observation(self) -> np.array:
        return self.observation_queue
    
    def step(self, action: int) -> tuple:
        total_reward = 0

        for _ in range(self.config.ENV.repeat):
            self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(action)
            total_reward += self.reward

        obs_tensor = torch.tensor(self.observation, dtype=torch.float32).permute(2, 0, 1)
        obs_transformed = self.transform(obs_tensor)
        obs_max = torch.maximum(self.previous_frame, obs_transformed)
        self.previous_frame = obs_transformed
        #wandb.log({"Examples": wandb.Image(obs_max.cpu().numpy())})
        self.observation_queue = torch.cat([self.observation_queue[1:], obs_max], dim=0)

        if self.config.ENV.reward_clip:
            self.reward =  max(-1, min(1, self.reward))

        return self.observation_queue, self.reward, self.terminated, self.truncated, self.info

    def quit(self) -> None:
        self.env.close()

    def screenshot(self, save:bool = True, file_path:str = 'figs/breakout_screenshot.png') -> None:
        plt.imshow(self.observation)

        if save:
            plt.savefig(file_path)

    def screenshot_of_convert(self, save:bool = True, file_path:str = 'figs/breakout_screenshot_converted.png') -> None:
        plt.imshow(self.transform(self.observation))

        if save:
            plt.savefig(file_path)

    def plot_rewards(self, bin_size:int = 1, show: bool = True, save: bool = True) -> None:
        rewards_rolling_average = np.convolve(self.rewards, np.ones(bin_size), 'valid') / bin_size
        
        plt.plot(rewards_rolling_average)
        plt.title('Breakout Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')

        if save:
            plt.savefig('figs/breakout_rewards.png')

        if show:
            plt.show()        
