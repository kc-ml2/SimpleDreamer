import torch
import torch.nn as nn
import numpy as np

from dreamer.algorithms.dreamer import Dreamer
from dreamer.modules.actor import Actor
from dreamer.modules.critic import Critic
from dreamer.modules.one_step_model import OneStepModel
from dreamer.utils.utils import (
    pixel_normalization,
    compute_lambda_values,
    create_normal_dist,
    DynamicInfos,
)
from dreamer.utils.buffer import ReplayBuffer


class Plan2Explore(Dreamer):
    def __init__(
        self,
        observation_shape,
        discrete_action_bool,
        action_size,
        writer,
        device,
        config,
    ):
        super().__init__(
            observation_shape, discrete_action_bool, action_size, writer, device, config
        )
        self.config = self.config + config.parameters.plan2explore

        self.intrinsic_actor = Actor(discrete_action_bool, action_size, config).to(
            self.device
        )
        self.intrinsic_critic = Critic(config).to(self.device)

        self.one_step_models = [
            OneStepModel(action_size, config).to(self.device)
            for _ in range(self.config.num_ensemble)
        ]
        self.one_step_models_params = nn.ModuleList(self.one_step_models).parameters()
        self.one_step_models_optimizer = torch.optim.Adam(
            self.one_step_models_params, lr=self.config.one_step_model_learning_rate
        )

        self.intrinsic_actor_optimizer = torch.optim.Adam(
            self.intrinsic_actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.intrinsic_critic_optimizer = torch.optim.Adam(
            self.intrinsic_critic.parameters(), lr=self.config.critic_learning_rate
        )

        self.intrinsic_actor.intrinsic = True
        self.actor.intrinsic = False

    def train(self, env):
        if len(self.buffer) < 1:
            self.environment_interaction(
                self.intrinsic_actor, env, self.config.seed_episodes
            )

        for iteration in range(self.config.train_iterations):
            for collect_interval in range(self.config.collect_interval):
                data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )
                posteriors, deterministics = self.dynamic_learning(data)
                self.behavior_learning(
                    self.actor,
                    self.critic,
                    self.actor_optimizer,
                    self.critic_optimizer,
                    posteriors,
                    deterministics,
                )

                self.behavior_learning(
                    self.intrinsic_actor,
                    self.intrinsic_critic,
                    self.intrinsic_actor_optimizer,
                    self.intrinsic_critic_optimizer,
                    posteriors,
                    deterministics,
                )

            self.environment_interaction(
                self.intrinsic_actor, env, self.config.num_interaction_episodes
            )
            self.evaluate(self.actor, env)

    def evaluate(self, actor, env):
        self.environment_interaction(actor, env, self.config.num_evaluate, train=False)

    def dynamic_learning(self, data):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))

        data.embedded_observation = self.encoder(data.observation)

        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1], deterministic
            )
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )

            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior

        infos = self.dynamic_learning_infos.get_stacked()
        self._model_update(data, infos)
        return infos.posteriors.detach(), infos.deterministics.detach()

    def _model_update(self, data, posterior_info):
        reconstructed_observation_dist = self.decoder(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            pixel_normalization(data.observation[:, 1:])
        )
        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
            continue_loss = self.continue_criterion(
                continue_dist.probs, 1 - data.done[:, 1:]
            )

        reward_dist = self.reward_predictor(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reward_loss = reward_dist.log_prob(data.reward[:, 1:])

        prior_dist = create_normal_dist(
            posterior_info.prior_dist_means,
            posterior_info.prior_dist_stds,
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info.posterior_dist_means,
            posterior_info.posterior_dist_stds,
            event_shape=1,
        )
        kl_divergence_loss = torch.mean(
            torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        )
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats).to(self.device), kl_divergence_loss
        )
        model_loss = (
            self.config.kl_divergence_scale * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )
        if self.config.use_continue_flag:
            model_loss += continue_loss.mean()

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.model_optimizer.step()

        predicted_feature_dists = [
            x(
                data.action[:, :-1],
                posterior_info.priors.detach(),
                posterior_info.deterministics.detach(),
            )
            for x in self.one_step_models
        ]
        one_step_model_loss = -sum(
            [
                x.log_prob(data.embedded_observation[:, 1:].detach()).mean()
                for x in predicted_feature_dists
            ]
        )

        self.one_step_models_optimizer.zero_grad()
        one_step_model_loss.backward()
        self.writer.add_scalar(
            "one step model loss", one_step_model_loss, self.num_total_episode
        )
        nn.utils.clip_grad_norm_(
            self.one_step_models_params,
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.one_step_models_optimizer.step()

    def behavior_learning(
        self, actor, critic, actor_optimizer, critic_optimizer, states, deterministics
    ):
        """
        #TODO : last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        state = states.reshape(-1, self.config.stochastic_size)
        deterministic = deterministics.reshape(-1, self.config.deterministic_size)

        # continue_predictor reinit
        for t in range(self.config.horizon_length):
            action = actor(state, deterministic)
            deterministic = self.rssm.recurrent_model(state, action, deterministic)
            _, state = self.rssm.transition_model(deterministic)
            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic, actions=action
            )

        self._agent_update(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            self.behavior_learning_infos.get_stacked(),
        )

    def _agent_update(
        self, actor, critic, actor_optimizer, critic_optimizer, behavior_learning_infos
    ):
        if actor.intrinsic:
            predicted_feature_means = [
                x(
                    behavior_learning_infos.actions,
                    behavior_learning_infos.priors,
                    behavior_learning_infos.deterministics
                ).mean
                for x in self.one_step_models
            ]
            predicted_feature_mean_stds = torch.stack(predicted_feature_means, 0).std(0)

            predicted_rewards = predicted_feature_mean_stds.mean(-1, keepdims=True)

        else:
            predicted_rewards = self.reward_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        values = critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean

        if self.config.use_continue_flag:
            continues = self.continue_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        else:
            continues = self.config.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            self.config.lambda_,
        )

        actor_loss = -torch.mean(lambda_values)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            actor.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        actor_optimizer.step()

        value_dist = critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))

        critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            critic.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        critic_optimizer.step()
    @torch.no_grad()
    def environment_interaction(self, actor, env, num_interaction_episodes, train=True):
        for epi in range(num_interaction_episodes):
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)
            action = torch.zeros(1, self.action_size).to(self.device)

            observation = env.reset()
            embedded_observation = self.encoder(
                torch.from_numpy(observation).float().to(self.device)
            )

            score = 0
            score_lst = np.array([])
            done = False

            while not done:
                deterministic = self.rssm.recurrent_model(
                    posterior, action, deterministic
                )
                embedded_observation = embedded_observation.reshape(1, -1)
                _, posterior = self.rssm.representation_model(
                    embedded_observation, deterministic
                )
                action = actor(posterior, deterministic).detach()

                if self.discrete_action_bool:
                    buffer_action = action.cpu().numpy()
                    env_action = buffer_action.argmax()

                else:
                    buffer_action = action.cpu().numpy()[0]
                    env_action = buffer_action

                next_observation, reward, done, info = env.step(env_action)
                if train:
                    self.buffer.add(
                        observation, buffer_action, reward, next_observation, done
                    )
                score += reward
                embedded_observation = self.encoder(
                    torch.from_numpy(next_observation).float().to(self.device)
                )
                observation = next_observation
                if done:
                    if train:
                        self.num_total_episode += 1
                        self.writer.add_scalar(
                            "training score", score, self.num_total_episode
                        )
                    else:
                        score_lst = np.append(score_lst, score)
                    break
        if not train:
            evaluate_score = score_lst.mean()
            print("evaluate score : ", evaluate_score)
            self.writer.add_scalar("test score", evaluate_score, self.num_total_episode)
