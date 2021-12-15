import numpy as np


def _preprocess_observation(obs):
    # return obs
    return obs["affordances"]  # (only valid for env=carla-*)


class StepSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = _preprocess_observation(self.env.reset())

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation
            action = policy(np.expand_dims(observation, 0), deterministic=deterministic)[0, :]

            next_observation, reward, done, _ = self.env.step(action)

            # we just want follow lane trajectories (only valid for env=carla-*)
            if next_observation["hlc"] != 3:
                self._traj_steps = 0
                self._current_observation = _preprocess_observation(self.env.reset())
            next_observation = _preprocess_observation(next_observation)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)

            self._current_observation = next_observation

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, action, reward, next_observation, done
                )

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = _preprocess_observation(self.env.reset())

        return dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )

    @property
    def env(self):
        return self._env


class TrajSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env

    def sample(self, policy, n_trajs, deterministic=False, replay_buffer=None, verbose=False):
        trajs = []
        info = {
            'collision': [],
            'out_of_lane': [],
            'mean_speed': [],
        }
        for n_traj in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []

            colision = False
            out_of_lane = False
            speeds = 0
            nb_steps = 0

            observation = _preprocess_observation(self.env.reset())

            for _ in range(self.max_traj_length):
                action = policy(np.expand_dims(observation, 0), deterministic=deterministic)[0, :]

                next_observation, reward, done, info_ = self.env.step(action)

                # # we just want follow lane trajectories (only valid for env=carla-*)
                if next_observation["hlc"] != 3:
                    break

                next_observation = _preprocess_observation(next_observation)
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_observations.append(next_observation)

                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, action, reward, next_observation, done
                    )

                observation = next_observation
                colision = colision or info_['colision']
                out_of_lane = out_of_lane or info_['out_of_lane']
                speeds += info_['speed']
                nb_steps += 1

                if done:
                    break

            trajs.append(dict(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
            ))
            info['collision'].append(colision)
            info['out_of_lane'].append(out_of_lane)
            info['mean_speed'].append(speeds / nb_steps)

            if verbose:
                print(f"Traj #{n_traj}, steps={nb_steps}, collision={colision}, return={np.sum(rewards):.3f}, "
                      f"out_of_lane={colision}, mean_speed={(speeds / nb_steps):.3f}")

        return trajs, info

    @property
    def env(self):
        return self._env
