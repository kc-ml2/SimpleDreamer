import collections
import collections.abc
for type_name in collections.abc.__all__:
        setattr(collections, type_name, getattr(collections.abc, type_name))

from attrdict import AttrDict
import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, observation_shape, action_size, device, config):
        self.config = config.parameters.dreamer.buffer
        self.device = device
        self.capacity = int(self.config.capacity)

        state_type = np.uint8 if len(observation_shape) < 3 else np.float32

        self.observation = np.empty(
            (self.capacity, *observation_shape), dtype=state_type
        )
        self.next_observation = np.empty(
            (self.capacity, *observation_shape), dtype=state_type
        )
        self.action = np.empty((self.capacity, action_size), dtype=np.float32)
        self.reward = np.empty((self.capacity, 1), dtype=np.float32)
        self.done = np.empty((self.capacity, 1), dtype=np.float32)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(self, observation, action, reward, next_observation, done):
        self.observation[self.buffer_index] = observation
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.next_observation[self.buffer_index] = next_observation
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size, chunk_size):
        """
        (batch_size, chunk_size, input_size)
        """
        last_filled_index = self.buffer_index - chunk_size + 1
        assert self.full or (
            last_filled_index > batch_size
        ), "too short dataset or too long chunk_size"
        sample_index = np.random.randint(
            0, self.capacity if self.full else last_filled_index, batch_size
        ).reshape(-1, 1)
        chunk_length = np.arange(chunk_size).reshape(1, -1)

        sample_index = (sample_index + chunk_length) % self.capacity

        observation = torch.as_tensor(
            self.observation[sample_index], device=self.device
        ).float()
        next_observation = torch.as_tensor(
            self.next_observation[sample_index], device=self.device
        ).float()

        action = torch.as_tensor(self.action[sample_index], device=self.device)
        reward = torch.as_tensor(self.reward[sample_index], device=self.device)
        done = torch.as_tensor(self.done[sample_index], device=self.device)

        sample = AttrDict(
            {
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "done": done,
            }
        )
        return sample
