import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from dreamer.utils.utils import create_normal_dist, build_network, horizontal_forward


class OneStepModel(nn.Module):
    def __init__(self, action_size, config):
        """
        For plan2explore
        There are several variations, but in our implementation,
        we use stochastic and deterministic actions as input and embedded observations as output
        """
        super().__init__()
        self.config = config.parameters.plan2explore.one_step_model
        self.embedded_state_size = config.parameters.dreamer.embedded_state_size
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.action_size = action_size

        self.network = build_network(
            self.deterministic_size + self.stochastic_size + action_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            self.embedded_state_size,
        )

    def forward(self, action, stochastic, deterministic):
        stoch_deter = torch.concat((stochastic, deterministic), axis=-1)
        x = horizontal_forward(
            self.network,
            action,
            stoch_deter,
            output_shape=(self.embedded_state_size,),
        )
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist
