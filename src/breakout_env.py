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

# housekeeping
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plt.style.use('ggplot')

class BreakoutEnvAgent():
    
    def __init__(self, render_mode: Union[str, None] = None, seed:int=0, plot_rewards_bool: bool = True) -> None:
        # MDP environment
        self.observation = None
        self.info = None
        self.reward = None
        self.terminated: bool = None
        self.truncated: bool = None

        # make the environment
        self._seed: int = seed
        self.render_mode: bool = render_mode

        # TODO: blurr with four frames
        self.observation_queue = [torch.zeros(3, 84, 84)]
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((84, 84)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

        # graphing vars
        self.plot_rewards_bool = plot_rewards_bool
        if plot_rewards_bool:
            self.reward_running: float = 0.0
            self.rewards: np.array = np.array([])
        

    def init_environment(self) -> None:
        self.env = gym.make(
                            'ALE/Breakout-v5', 
                            render_mode=self.render_mode
                            )
        self.env.seed(self.seed)
        self.observation, self.info = self.env.reset(seed=self.seed)

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, seed:int) -> None:
        self.env.seed(seed)

    def reset(self, seed:int=0) -> tuple:
        if self.plot_rewards_bool:
            self.rewards = np.append(self.rewards, self.reward_running)
            self.reward_running = 0.0
        self.terminated = False
        self.truncated = False
        self.observation, self.info = self.env.reset(seed=seed)
        return self.observation, self.info

    def get_action_space(self) -> gym.spaces.Discrete:
        return self.env.action_space

    def get_observation_space(self) -> gym.spaces.Box:
        return self.env.observation_space
    
    def get_reward_range(self) -> tuple:
        return self.env.reward_range
    
    @classmethod
    def convert_to_grayscale(self, element) -> np.array:
        return np.dot(element, [0.299, 0.587, 0.114])
    
    def convert_observation(self) -> np.array:
        if len(self.observation_queue) < 4:
            return self.observation_queue[0]
        else:
            return (self.observation_queue[0] +
                    self.observation_queue[1] +
                    self.observation_queue[2] +
                    self.observation_queue[3]
                )/4.0
    
    def step(self, action:int) -> tuple:
        self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(action)
        
        # update observation queue
        if len(self.observation_queue) < 4:
            self.observation_queue.append(self.transform(self.observation))
        else:
            self.observation_queue.pop(0)
            self.observation_queue.append(self.transform(self.observation))

        self.reward_running += self.reward
        return self.observation, self.reward, self.terminated, self.truncated, self.info
    
    def quit(self) -> None:
        self.env.close()

    def screenshot(self, save:bool = True, file_path:str = 'figs/breakout_screenshot.png') -> None:
        plt.imshow(self.observation)

        if save:
            plt.savefig(file_path)

    def screenshot_of_convert(self, save:bool = True, file_path:str = 'figs/breakout_screenshot_converted.png') -> None:
        plt.imshow(self.convert_observation())

        if save:
            plt.savefig(file_path)

    #TODO: create gif from screenshots

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

class BreakoutEnvPlayer():

    def __init__(self, render_mode:str='rgb_array') -> None:
        self.render_mode = render_mode

    def init_environment(self) -> None:
        play(gym.make(
                    'ALE/Breakout-v5', 
                    render_mode=self.render_mode
                    ))


def main():
    env = BreakoutEnvAgent(
        render_mode='human'
    )

    env.init_environment()

    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.get_action_space().sample())
        if terminated:
            break

if __name__ == "__main__":
    main()