"""Dummy environment code based on 
https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952"""

import gym
from gym import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(2)

    def step(self, action):
        state = 1

        if action == 2:
            reward = 1
        else:
            reward = -1

        done = True
        info = {}
        return state, reward, done, info

    def reset(self):
        state = 0
        return state
