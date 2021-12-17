import numpy as np
import torch

def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }

# TODO: 4 buffer, one for each hlc
class ReplayBuffer(object):
    def __init__(self, max_size):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0

    def __len__(self):
        return self._size

    def _init_storage(self, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._dones = np.zeros(self._max_size, dtype=np.float32)
        self._hlcs = np.zeros(self._max_size, dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add_sample(self, observation, action, reward, next_observation, done, hlc):
        if not self._initialized:
            self._init_storage(observation.size, action.size)

        self._observations[self._next_idx, :] = np.array(observation, dtype=np.float32)
        self._next_observations[self._next_idx, :] = np.array(next_observation, dtype=np.float32)
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = float(done)
        self._hlcs[self._next_idx] = float(hlc)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones, hlcs):
        for o, a, r, no, d, h in zip(observations, actions, rewards, next_observations, dones, hlcs):
            self.add_sample(o, a, r, no, d, h)

    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        return dict(
            observations=self._observations[indices, ...],
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=self._next_observations[indices, ...],
            dones=self._dones[indices, ...],
            hlcs=self._hlcs[indices, ...],
        )

    def torch_sample(self, batch_size, device):
        return batch_to_torch(self.sample(batch_size), device)

    def sample_generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    def torch_sample_generator(self, batch_size, device, n_batchs=None):
        for batch in self.sample_generator(batch_size, n_batchs):
            yield batch_to_torch(batch, device)

    @property
    def total_steps(self):
        return self._total_steps


class ReplayBufferHLC(object):
    def __init__(self, max_size, nb_hlc=4):
        self._buffers = [ReplayBuffer(max_size) for _ in range(nb_hlc)]
        self._nb_hlc = nb_hlc
        self._total_steps = np.sum([buffer.total_steps() for buffer in self._buffers])

    def __len__(self):
        return np.sum([len(buffer) for buffer in self._buffers])

    def add_sample(self, observation, action, reward, next_observation, done, hlc):
        for i, buffer in enumerate(self._buffers):
            f_observation = observation[hlc == i]
            f_action = action[hlc == i]
            f_reward = reward[hlc == i]
            f_next_observation = next_observation[hlc == i]
            f_done = done[hlc == i]
            f_hlc = hlc[hlc == i]
            buffer.add_sample(
                f_observation,
                f_action,
                f_reward,
                f_next_observation,
                f_done,
                f_hlc,
            )
    
    def add_traj(self, observations, actions, rewards, next_observations, dones, hlcs):
        for i, buffer in enumerate(self._buffers):
            f_observations = observations[hlcs == i]
            f_actions = actions[hlcs == i]
            f_rewards = rewards[hlcs == i]
            f_next_observations = next_observations[hlcs == i]
            f_dones = dones[hlcs == i]
            f_hlcs = hlcs[hlcs == i]
            buffer.add_traj(
                f_observations,
                f_actions,
                f_rewards,
                f_next_observations,
                f_dones,
                f_hlcs,
            )
    
    def sample(self, batch_size):
        samples = {}
        for buffer in self._buffers:
            for key, value in buffer.sample(batch_size):
                if key in samples:
                    samples[key] = np.concatenate((value, samples[key]), axis=0)
                else:
                    samples[key] = value
        return samples

    def torch_sample(self, batch_size, device):
        return batch_to_torch(self.sample(batch_size), device)

    def sample_generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    def torch_sample_generator(self, batch_size, device, n_batchs=None):
        for batch in self.sample_generator(batch_size, n_batchs):
            yield batch_to_torch(batch, device)

    @property
    def total_steps(self):
        return self._total_steps