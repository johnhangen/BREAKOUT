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


class BreakoutEnvAgent():
    
    def __init__(self, render_mode:str='human', seed:int=0, plot_rewards_bool: bool = True) -> None:
        # MDP environment
        self.observation = None
        self.info = None
        self.reward = None
        self.terminated: bool = None
        self.truncated: bool = None

        # make the environment
        self._seed = seed
        self.render_mode = render_mode

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
    
    def convert_observation(self) -> np.array:
        return np.dot(self.observation, [0.299, 0.587, 0.114]).flatten()
    
    def step(self, action:int) -> tuple:
        self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(action)
        self.reward_running += self.reward
        return self.observation, self.reward, self.terminated, self.truncated, self.info
    
    def quit(self) -> None:
        self.env.close()

    def plot_rewards(self, show: bool = True, save: bool = True) -> None:
        plt.plot(self.rewards)
        plt.title('Breakout Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')

        if save:
            plt.savefig('breakout_rewards.png')

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
    env = BreakoutEnvAgent()

    env.init_environment()

    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.get_action_space().sample())
        if terminated:
            break

if __name__ == "__main__":
    main()