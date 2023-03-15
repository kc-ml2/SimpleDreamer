import gym
import numpy as np


class ChannelFirstEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.observation_space
        obs_shape = obs_space.shape[-1:] + obs_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        return observation


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
