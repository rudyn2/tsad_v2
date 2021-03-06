import argparse
import pprint

import torch

from sampler import TrajSampler
from src.utils.utils import Timer
from src.models.model import SamplerPolicy
from src.models.model import TanhGaussianPolicy
from gym_carla.envs.carla_pid_env import CarlaPidEnv
from gym_carla.envs.carla_env import CarlaEnv
import numpy as np


ENV_PARAMS = {
            # carla connection parameters+
            'host': 'localhost',
            'port': 2000,  # connection port
            'town': 'Town01',  # which town to simulate
            'traffic_manager_port': 8000,

            # simulation parameters
            'verbose': False,
            'vehicles': 100,  # number of vehicles in the simulation
            'walkers': 10,  # number of walkers in the simulation
            'obs_size': 224,  # sensor width and height
            'max_past_step': 1,  # the number of past steps to draw
            'dt': 0.025,  # time interval between two frames
            'normalized_input': True,
            'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
            'max_time_episode': 500,  # maximum timesteps per episode
            'max_waypt': 12,  # maximum number of waypoints
            'd_behind': 12,  # distance behind the ego vehicle (meter)
            'out_lane_thres': 2.0,  # threshold for out of lane
            'desired_speed': 6,  # desired speed (m/s)
            'speed_reduction_at_intersection': 0.75,
            'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        }


def main(eval_variant):
    env_params = eval_variant["env_params"]
    if variant["act_mode"] == "pid":
        env_params.update({
            'continuous_speed_range': [0.0, env_params["desired_speed"]],
            'continuous_steer_range': [-1.0, 1.0],
        })
        carla_env = CarlaPidEnv(env_params)
    else:
        env_params.update({
            'continuous_throttle_range': [0.0, 1.0],
            'continuous_brake_range': [0.0, 1.0],
            'continuous_steer_range': [-1.0, 1.0],
        })
        carla_env = CarlaEnv(env_params)

    eval_sampler = TrajSampler(carla_env, eval_variant["max_traj_length"])

    policy = TanhGaussianPolicy(
        carla_env.observation_space.shape[0],
        carla_env.action_space.shape[0],
        eval_variant["policy_arch"],
        log_std_multiplier=eval_variant["policy_log_std_multiplier"],
        log_std_offset=eval_variant["policy_log_std_offset"],
    )
    policy.load_state_dict(torch.load(eval_variant["checkpoint"]))
    policy.to(eval_variant["device"])
    sampler_policy = SamplerPolicy(policy, eval_variant["device"])

    metrics = {}
    print("Evaluating in the environment")
    with Timer() as eval_timer:
        trajs = eval_sampler.sample(
            sampler_policy, eval_variant["eval_n_trajs"], deterministic=True
        )
        metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
        metrics['max_return'] = np.max([np.sum(t['rewards']) for t in trajs])
        metrics['min_return'] = np.min([np.sum(t['rewards']) for t in trajs])
        metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])

        # calculate weighted average return
        total_lengths = np.sum([t["rewards"].shape[0] for t in trajs])
        weighted_return = 0
        for t in trajs:
            weighted_return += (t['rewards'].shape[0] / total_lengths) * np.sum(t['rewards'])
        metrics['average_weighted_return'] = weighted_return

    metrics['eval_time'] = eval_timer()
    pprint.pprint(metrics)


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        max_traj_length=500,
        seed=42,
        device='cuda',
        act_mode='pid',

        policy_arch='256-256',
        qf_arch='256-256',
        policy_log_std_multiplier=1.0,
        policy_log_std_offset=-1.0,
        eval_n_trajs=10,
        env_params=ENV_PARAMS
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint.")
    args = parser.parse_args()
    variant["checkpoint"] = args.checkpoint
    main(variant)
