from ml_collections import ConfigDict
import torch
import torch.nn.functional as F


def soft_target_update(network, target_network, soft_target_update_rate):
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            (1 - soft_target_update_rate) * target_network_params[k].data
            + soft_target_update_rate * v.data
        )

class DDPG(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.reward_scale = 1.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.policy_frequency = 2
        config.max_grad_norm = 0.5
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3

        if updates is not None:
            config.update(updates)
        return config

    def __init__(self, config, policy, target_policy, qf1, target_qf1):
        self.config = DDPG.get_default_config(config)
        self.policy = policy
        self.target_policy = target_policy
        self.qf1 = qf1
        self.target_qf1 = target_qf1

        optimizer_class = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), self.config.policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf1.parameters(), self.config.qf_lr
        )
        self.update_target_network(1.0)
        self._total_steps = 0

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.policy, self.target_policy, soft_target_update_rate)

    def train(self, batch):
        self._total_steps += 1

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']
        hlcs = batch['hlcs']

        """ Q function loss """
        with torch.no_grad():
            target_q_values = self.target_qf1(next_observations, self.target_policy(next_observations, hlcs), hlcs)
            q_target = self.config.reward_scale * rewards + (1. - dones) * self.config.discount * target_q_values
        q1_pred = self.qf1(observations, actions, hlcs)
        qf1_loss = F.mse_loss(q1_pred, q_target)

        self.qf_optimizer.zero_grad()
        qf1_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.qf1.parameters()), self.config.max_grad_norm)
        self.qf_optimizer.step()

        """ Policy loss """
        policy_loss = None
        if self._total_steps % self.config.policy_frequency == 0:
            policy_loss = -self.qf1(observations, self.policy(observations, hlcs), hlcs).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()), self.config.max_grad_norm)
            self.policy_optimizer.step()

            # update the target networks
            self.update_target_network(self.config.soft_target_update_rate)

        ddpg_metrics = dict(
            qf1_loss=qf1_loss.item(),
            average_qf1=q1_pred.mean().item(),
            average_target_q=target_q_values.mean().item(),
            total_steps=self.total_steps
        )
        if policy_loss is not None:
            ddpg_metrics["policy_loss"] = policy_loss.item()

        batch_metrics = dict(
            reward_min=rewards.min().item(),
            reward_max=rewards.max().item(),
            rewards_mean=rewards.mean().item()
        )
        return ddpg_metrics, batch_metrics

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.policy, self.target_policy, self.qf1, self.target_qf1]
        return modules

    @property
    def total_steps(self):
        return self._total_steps
