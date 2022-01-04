import numpy as np
import torch
import random
from collections import deque, namedtuple

def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }

class ReplayBuffer(object):
    def __init__(self, max_size):
        self._memory = deque([], maxlen=max_size)
        self._max_size = max_size
        self._empty_transition = {
            "observations": np.array([]),
            "next_observations": np.array([]),
            "actions": np.array([]),
            "rewards": np.array([]),
            "dones": np.array([]),
            "hlcs": np.array([]),
        }
        self._Transition = namedtuple(
            "Transition", tuple(self._empty_transition.keys())
        )
        self._total_steps = 0

    def empty(self):
        """Empty memory"""        
        self._memory.clear()

    def push(self, *args):
        """Save a experiences"""
        self._memory.append(self._Transition(*args))
    
    def __len__(self):
        return len(self._memory)

    def add_sample(self, observation, action, reward, next_observation, done, hlc):
        self.push(
            np.array(observation, dtype=np.float32)[np.newaxis,:],
            np.array(next_observation, dtype=np.float32)[np.newaxis,:],
            np.array(action, dtype=np.float32)[np.newaxis,:],
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=np.float32),
            np.array([hlc], dtype=np.float32),
        )
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones, hlcs):
        for o, a, r, no, d, h in zip(observations, actions, rewards, next_observations, dones, hlcs):
            self.add_sample(o, a, r, no, d, h)

    def sample(self, batch_size):
        size = min(len(self), batch_size)
        if size == 0:
            sample = self._empty_transition
        else:
            sample = random.sample(self._memory, size)
            sample = self._Transition(*zip(*sample))
            sample = self._unpack(sample)
        return sample
    
    def _unpack(self, sample):
        unpacked = {}
        for name, value in sample._asdict().items():
            unpacked[name] = np.concatenate(value, axis=0)
        return unpacked

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
        self._total_steps = np.sum([buffer.total_steps for buffer in self._buffers])

    def __len__(self):
        return np.sum([len(buffer) for buffer in self._buffers])

    def add_sample(self, observation, action, reward, next_observation, done, hlc):
        self._buffers[hlc].add_sample(
                observation,
                action,
                reward,
                next_observation,
                done,
                hlc,
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
            for key, value in buffer.sample(batch_size).items():
                if key not in samples or len(samples[key].shape) == 0 or samples[key].shape[0] == 0:
                    samples[key] = value
                elif len(value.shape) > 0:
                    # If key is already in dict and samples[key] has length and value has length
                    samples[key] = np.concatenate((value, samples[key]), axis=0)
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