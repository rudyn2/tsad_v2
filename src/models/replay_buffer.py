import glob
import os
import numpy as np
import torch
import random
import json
from collections import deque, namedtuple, defaultdict


def batch_to_torch(batch, device):
    return {
        hlc: {
            k: torch.from_numpy(v).to(device=device, non_blocking=True)
            for k, v in hlc_batch.items()
        } for hlc, hlc_batch in batch.items()
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
            "dones": np.array([])
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

    def add_sample(self, observation, action, reward, next_observation, done):
        self.push(
            np.array(observation, dtype=np.float32)[np.newaxis, :],
            np.array(next_observation, dtype=np.float32)[np.newaxis, :],
            np.array(action, dtype=np.float32)[np.newaxis, :],
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=np.float32)
        )
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.add_sample(o, a, r, no, d)

    def sample(self, batch_size):
        if len(self) < batch_size:
            return self._empty_transition
        else:
            sample = random.sample(self._memory, batch_size)
            sample = self._Transition(*zip(*sample))
            return self._unpack(sample)

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
    def __init__(self, max_size, hlcs=(0, 1, 2, 3)):
        self._buffers = {str(hlc): ReplayBuffer(max_size) for hlc in hlcs}
        self._hlcs = hlcs
        self._total_steps = np.sum([buffer.total_steps for buffer in self._buffers.values()])

    def __len__(self):
        return np.sum([len(buffer) for buffer in self._buffers.values()])

    def add_sample(self, observation, action, reward, next_observation, done, hlc):
        self._buffers[str(hlc)].add_sample(
                observation,
                action,
                reward,
                next_observation,
                done
            )

    # IS THIS METHOD DEPRECATED?
    def add_traj(self, observations, actions, rewards, next_observations, dones, hlcs):
        for i, buffer in self._buffers.items():
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
        return {hlc: buffer.sample(batch_size) for hlc, buffer in self._buffers.items()}

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


class BCDatasetHLC(object):
    def __init__(self, data_path: str, hlcs=(0, 1, 2, 3)):
        self._data_path = data_path
        self._hlcs = hlcs
        self._data = self.read()

    def read(self) -> dict:
        """
        :return: {
            str(hlc): {
                'observations': np.array(...),
                'actions': np.array(...)
            }
         }
        """
        json_files = glob.glob(os.path.join(self._data_path, "*.json"))
        all_data = {str(hlc): dict(observations=[], actions=[]) for hlc in self._hlcs}
        for json_file in json_files:
            with open(json_file, "r") as f:
                batch_data = json.load(f)
            i = 0
            for ep_key, ep_values in batch_data.items():
                i += 1
                print(f"Loading episode: {ep_key} ({i}/{len(batch_data)})")
                timestamps = list(ep_values.keys())
                for idx in range(len(timestamps) - 3):
                    ts = timestamps[idx]
                    ts_value = ep_values[ts]
                    rl_hlc = ts_value["command"] - 1
                    if rl_hlc in self._hlcs:
                        steer = ts_value["rl"]["action"][2]    # recover action steering
                        speed = ep_values[timestamps[idx + 3]]["speed"]
                        all_data[str(rl_hlc)]['observations'].append(np.array(ts_value["rl"]["obs"]))
                        all_data[str(rl_hlc)]['actions'].append(np.array([speed, steer]))

        for hlc in self._hlcs:
            all_data[str(hlc)]['observations'] = np.array(all_data[str(hlc)]['observations']).astype(np.float32)
            all_data[str(hlc)]['actions'] = np.array(all_data[str(hlc)]['actions']).astype(np.float32)
        return all_data

    def sample(self, batch_size: int):
        return {str(hlc): self.sample_hlc(batch_size, hlc) for hlc in self._hlcs}

    def sample_hlc(self, batch_size: int, hlc: int):
        available_idxs = list(range(len(self._data[str(hlc)]['observations'])))
        idxs = random.sample(available_idxs, batch_size)
        return {
            'observations': np.array(self._data[str(hlc)]['observations'][idxs]),
            'actions': np.array(self._data[str(hlc)]['actions'][idxs])
        }

    def len(self):
        return sum([v['observations'].shape[0] for v in self._data.values()])


if __name__ == "__main__":
    dataset = BCDatasetHLC(data_path="/home/rudy/Downloads/affordances")
    sample = dataset.sample(128)
    # pseudo-test
    buffer = ReplayBufferHLC(max_size=10000, hlcs=(0, 1, 2, 3))
    for _ in range(10):
        buffer.add_sample(
            observation=np.random.rand(3),
            action=np.random.rand(2),
            reward=np.random.rand(),
            next_observation=np.random.rand(3),
            done=random.choice([True, False]),
            hlc=random.choice([0, 1, 2, 3])
        )
    samples = buffer.sample(128)


