import torch
import torch.nn as nn
from dreamer.utils.utils import build_network, create_normal_dist, horizontal_forward


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.agent.critic
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            1,
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=(1,)
        )
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist
