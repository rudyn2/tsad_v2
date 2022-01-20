import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
# from torch.distributions.transforms import TanhTransform
from src.utils.eps_scheduler import Epsilon
from src.utils.transforms import TanhTransform


class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, arch='256-256'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch

        d = input_dim
        modules = []
        hidden_sizes = [int(h) for h in arch.split('-')]

        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size

        last_fc = nn.Linear(d, output_dim)
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)


class ReparameterizedTanhGaussian(nn.Module):

    def __init__(self, log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, mean, log_std, deterministic=False):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        action_distribution = TransformedDistribution(
            Normal(mean, std), TanhTransform(cache_size=1)
        )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(
            action_distribution.log_prob(action_sample), dim=1
        )

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256',
                 log_std_multiplier=1.0, log_std_offset=-1.0):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch

        self.base_network = FullyConnectedNetwork(
            observation_dim, 2 * action_dim, arch
        )
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian()

    def forward(self, observations, deterministic=False):
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian(mean, log_std, deterministic)


class SamplerPolicy(object):

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def __call__(self, observations, deterministic=False):
        with torch.no_grad():
            single_action = len(observations.shape) == 1
            new_observations = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )

            if single_action:
                new_observations = new_observations.unsqueeze(0)

            actions, _ = self.policy(new_observations, deterministic)

            if single_action:
                actions = actions.squeeze(0)
            actions = actions.cpu().numpy()
        return actions


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )

    def forward(self):
        return self.constant


class DDPGSamplerPolicy(object):

    def __init__(self, policy, device, exploration_noise=0.1, max_steps=10000, action_low=-1, action_max=1):
        self.policy = policy
        self.device = device
        self.exploration_noise = exploration_noise
        self.eps_scheduler = Epsilon(max_steps, epsilon_max=0.5, epsilon_min=0.05)
        self.action_low = action_low
        self.action_max = action_max

    def __call__(self, observations, hlc, deterministic=False):
        with torch.no_grad():
            observations = torch.from_numpy(observations).float().to(self.device)
            if len(observations.shape) == 1:
                observations = observations.unsqueeze(0)
            actions = self.policy(observations, hlc, deterministic=deterministic)
            if not deterministic:
                noise = torch.normal(0, self.eps_scheduler.step(), size=actions.shape, device=self.device)
                actions = torch.clamp(actions + noise, self.action_low, self.action_max)
            if len(observations.shape) == 1 or observations.shape[0] == 1:
                actions = actions.squeeze(0)
            actions = actions.cpu().numpy()
        return actions

    def get_epsilon(self):
        return self.eps_scheduler.epsilon()


class FullyConnectedQFunction(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256'):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.network = FullyConnectedNetwork(
            observation_dim + action_dim, 1, arch
        )

    def forward(self, observations, actions):
        input_tensor = torch.cat([observations, actions], dim=1)
        return torch.squeeze(self.network(input_tensor), dim=1)


class FullyConnectedQFunctionHLC(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256', hlcs=(0, 1, 2, 3)):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.hlcs = hlcs
        self.networks = nn.ModuleDict({str(hlc): FullyConnectedQFunction(observation_dim, action_dim, arch)
                                       for hlc in hlcs})

    def forward(self, observations, actions, hlc):
        output = self.networks[str(hlc)](observations, actions)
        return output

    def transfer_learning(self, from_hlc: int = 3, to_hlcs: tuple = (0, 1, 2)):
        if from_hlc in self.hlcs:
            for hlc in to_hlcs:
                if hlc in self.hlcs:
                    self.networks[str(hlc)].load_state_dict(self.networks[str(from_hlc)].state_dict())
                    print(f"Transferred from {from_hlc} to {hlc}")
        else:
            print("Origin network is not present in the policy.")


class FullyConnectedTanhPolicy(nn.Module):
    def __init__(self, observation_dim, action_dim, arch='256-256'):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.network = FullyConnectedNetwork(
            observation_dim, action_dim, arch
        )

    # deterministic parameter just for compatibility
    def forward(self, observation, deterministic=True):
        output = self.network(observation)
        output = torch.tanh(output)
        return output


class FullyConnectedTanhPolicyHLC(nn.Module):
    def __init__(self, observation_dim, action_dim, arch='256-256', hlcs=(0, 1, 2, 3)):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.hlcs = hlcs
        self.networks = nn.ModuleDict({str(hlc): FullyConnectedTanhPolicy(
            observation_dim, action_dim, arch
        ) for hlc in hlcs})

    # deterministic parameter just for compatibility
    def forward(self, observation, hlc, deterministic=True):
        prediction = self.networks[str(hlc)](observation)
        return prediction

    def transfer_learning(self, from_hlc: int = 3, to_hlcs: tuple = (0, 1, 2)):
        if from_hlc in self.hlcs:
            for hlc in to_hlcs:
                if hlc in self.hlcs:
                    self.networks[str(hlc)].load_state_dict(self.networks[str(from_hlc)].state_dict())
                    print(f"Transferred from {from_hlc} to {hlc}")
        else:
            print("Origin network is not present in the policy.")
