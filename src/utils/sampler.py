import numpy as np
import carla


def _preprocess_observation(obs):
    # return obs
    return obs["affordances"], obs["hlc"]  # (only valid for env=carla-*)


class StepSampler(object):

    def __init__(self, env, allowed_hlcs, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self.allowed_hlcs = allowed_hlcs
        self._traj_steps = 0
        self._current_observation, self._current_hlc = _preprocess_observation(self.env.reset())

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None, draw_waypoints=False):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        hlcs = []

        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation
            hlc = self._current_hlc
            action = policy(np.expand_dims(observation, 0), hlc=hlc, deterministic=deterministic)
            
            next_observation, reward, done, _ = self.env.step(action)
            next_observation, next_hlc = _preprocess_observation(next_observation)

            if next_hlc not in self.allowed_hlcs:
                self._traj_steps = 0
                self._current_observation, self._current_hlc = _preprocess_observation(self.env.reset())
                break

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)
            hlcs.append(hlc)

            self._current_observation = next_observation
            self._current_hlc = next_hlc

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, action, reward, next_observation, done, hlc
                )

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation, self._current_hlc = _preprocess_observation(self.env.reset())

        return dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
            hlcs=np.array(hlcs, dtype=np.float32)
        )

    @property
    def env(self):
        return self._env


class TrajSampler(object):

    def __init__(self, env, allowed_hlcs, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._allowed_hlcs = allowed_hlcs
        self._env = env

    def sample(self, policy, n_trajs, deterministic=False, replay_buffer=None, verbose=False, draw_waypoints=False):
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
            hlcs = []

            colision = False
            out_of_lane = False
            speeds = 0
            nb_steps = 0

            observation, hlc = _preprocess_observation(self.env.reset())

            for _ in range(self.max_traj_length):
                action = policy(np.expand_dims(observation, 0), hlc=hlc, deterministic=deterministic)

                next_observation, reward, done, info_ = self.env.step(action)

                if draw_waypoints:
                    for waypoint in info_['waypoints']:
                        self._env.world.debug.draw_string(carla.Location(x=waypoint[0], y=waypoint[1], z=0),
                                                          'x',
                                                          draw_shadow=False,
                                                          color=carla.Color(r=255, g=0, b=0),
                                                          life_time=5,
                                                          persistent_lines=True)

                next_observation, next_hlc = _preprocess_observation(next_observation)

                if next_hlc not in self._allowed_hlcs:
                    break

                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_observations.append(next_observation)
                hlcs.append(hlc)

                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, action, reward, next_observation, done, hlc
                    )

                observation = next_observation
                hlc = next_hlc
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
                hlcs=np.array(hlcs, dtype=np.float32)
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
