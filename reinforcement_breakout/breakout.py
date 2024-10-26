
import collections
import cv2
import gym
import numpy as np
from PIL import Image
import torch


class DQNBreakout(gym.Wrapper):

    def __init__(self, render_mode='rgb_array', repeat=4, device=torch.device('cpu')):
        env = gym.make('BreakoutNoFrameskip-v4', render_mode=render_mode)

        super(DQNBreakout, self).__init__(env)
        self.repeat=repeat
        self.lives = env.unwrapped.ale.lives()

    def step(self, action):
        """overwriting default step function
        making simplified step function without truncated"""

        total_reward = 0
        done = False

        for i in range(self.repeat):
            """means our worker will take the same action for at least 4 (repeat value) frames"""
            observation, reward, done, truncated, info = self.env.step(action)

            total_reward += reward