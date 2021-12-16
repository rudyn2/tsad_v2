import os
import sys
from copy import deepcopy
import argparse
import numpy as np
import carla
import gym
from gym_carla.envs.carla_env import CarlaEnv
from gym_carla.envs.carla_pid_env import CarlaPidEnv

from ddpg import DDPG
from src.models.replay_buffer import ReplayBuffer, batch_to_torch
from src.models.model import FullyConnectedQFunction, DDPGSamplerPolicy, FullyConnectedTanhPolicy
from src.utils.sampler import StepSampler, TrajSampler
from src.utils.utils import Timer, set_random_seed, prefix_metrics
from src.utils.utils import WandBLogger


ENV_PARAMS = {
            # carla connection parameters+
            'host': 'localhost',
            'port': 2000,  # connection port
            'town': 'Town01',  # which town to simulate
            'traffic_manager_port': 8000,

            # simulation parameters
            'verbose': False,
            'vehicles': 50,  # number of vehicles in the simulation
            'walkers': 10,  # number of walkers in the simulation
            'obs_size': 224,  # sensor width and height
            'max_past_step': 1,  # the number of past steps to draw
            'dt': 0.025,  # time interval between two frames
            'normalized_input': True,
            'max_time_episode': 800,  # maximum timesteps per episode
            'max_waypt': 12,  # maximum number of waypoints
            'd_behind': 10,  # distance behind the ego vehicle (meter)
            'out_lane_thres': 2.0,  # threshold for out of lane
            'desired_speed': 6,  # desired speed (m/s)
            'speed_reduction_at_intersection': 0.75,
            'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        }


def main(variant):
    wandb_logger = WandBLogger(config=WandBLogger.get_default_config(), variant=variant)
    wandb_config = wandb_logger.wandb_config
    set_random_seed(wandb_config["seed"])

    # integrate gym-carla here
    env_params = wandb_config["env_params"]
    if variant["env"] == "carla-pid":
        env_params.update({
            'continuous_speed_range': [0.0, env_params["desired_speed"]],
            'continuous_steer_range': [-1.0, 1.0],
        })
        env = CarlaPidEnv(env_params)
        action_low, action_max = -1, 1
    elif variant["env"] == "carla-normal":
        env_params.update({
            'continuous_throttle_range': [0.0, 1.0],
            'continuous_brake_range': [0.0, 1.0],
            'continuous_steer_range': [-1.0, 1.0],
        })
        action_low, action_max = -1, 1
        env = CarlaEnv(env_params)
    else:
        action_low, action_max = -2, 2
        env = gym.make(variant["env"])

    train_sampler = StepSampler(env, wandb_config["max_traj_length"])
    eval_sampler = TrajSampler(env, wandb_config["max_traj_length"])

    replay_buffer = ReplayBuffer(wandb_config["replay_buffer_size"])

    policy = FullyConnectedTanhPolicy(
        train_sampler.env.observation_space.shape[0],
        train_sampler.env.action_space.shape[0],
        wandb_config["policy_arch"]
    )
    target_policy = deepcopy(policy)

    qf1 = FullyConnectedQFunction(
        train_sampler.env.observation_space.shape[0],
        train_sampler.env.action_space.shape[0],
        wandb_config["qf_arch"]
    )
    target_qf1 = deepcopy(qf1)

    ddpg = DDPG(wandb_config["ddpg"], policy, target_policy, qf1, target_qf1)
    ddpg.torch_to_device(wandb_config["device"])

    sampler_policy = DDPGSamplerPolicy(policy,
                                       wandb_config["device"],
                                       max_steps=wandb_config["noise_max_steps"],
                                       action_low=action_low,
                                       action_max=action_max)

    print("Training...")
    max_return = 0
    for epoch in range(wandb_config['n_epochs']):
        metrics = {}
        sys.stdout.flush()
        sys.stdout.write(f"rolling epoch={epoch}".ljust(100))
        sys.stdout.write("\r")
        with Timer() as rollout_timer:
            train_sampler.sample(
                sampler_policy, wandb_config["n_env_steps_per_epoch"],
                deterministic=False, replay_buffer=replay_buffer
            )
            metrics['epsilon'] = sampler_policy.get_epsilon()
            metrics['env_steps'] = replay_buffer.total_steps
            metrics['epoch'] = epoch

        sys.stdout.flush()
        sys.stdout.write(f"training epoch={epoch}".ljust(100))
        sys.stdout.write("\r")
        with Timer() as train_timer:
            for batch_idx in range(wandb_config["n_train_step_per_epoch"]):
                batch = batch_to_torch(replay_buffer.sample(wandb_config["batch_size"]), wandb_config["device"])
                if batch_idx + 1 == wandb_config["n_train_step_per_epoch"]:
                    ddpg_metrics, batch_metrics = ddpg.train(batch)
                    metrics.update(prefix_metrics(ddpg_metrics, 'ddpg'))
                    metrics.update(prefix_metrics(batch_metrics, 'batch'))
                else:
                    ddpg.train(batch)

        sys.stdout.flush()
        sys.stdout.write(f"evaluating epoch={epoch}".ljust(100))
        sys.stdout.write("\r")
        with Timer() as eval_timer:
            metrics["average_weighted_return"] = 0
            if epoch == 0 or (epoch + 1) % wandb_config["eval_period"] == 0:
                trajs, info = eval_sampler.sample(
                    sampler_policy, wandb_config["eval_n_trajs"], deterministic=True
                )
                metrics['collision_rate'] = np.mean(info['collision'])
                metrics['out_of_lane_rate'] = np.mean(info['out_of_lane'])
                metrics['mean_speed'] = np.mean(info['mean_speed'])
                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                total_lengths = np.sum([t["rewards"].shape[0] for t in trajs])
                weighted_return = 0
                for t in trajs:
                    weighted_return += (t['rewards'].shape[0] / total_lengths) * np.sum(t['rewards'])
                metrics['average_weighted_return'] = weighted_return

        metrics['rollout_time'] = rollout_timer()
        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = rollout_timer() + train_timer() + eval_timer()

        print(f"Epoch {epoch}, rollout_time={metrics['rollout_time']:.0f}s, train_time={metrics['train_time']:.0f}s, "
              f"eval_time={metrics['eval_time']:.0f}s, epoch_time={metrics['epoch_time']:.0f}s")

        # checkpoint condition
        if metrics["average_weighted_return"] > max_return:
            print(f"Saving model at epoch {epoch} [max_return={max_return:.2f}].")
            max_return = metrics["average_weighted_return"]
            wandb_logger.save_models(policy, target_qf1, tag='best')
        wandb_logger.log(metrics)
    wandb_logger.save_models(policy, target_qf1, tag='last')
    print("Done!")


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        max_traj_length=1000,
        replay_buffer_size=1000000,
        seed=42,
        device='cuda',
        env='carla-pid',    # carla-pid, carla-normal, Pendulum-v0

        policy_arch='256-256-256',
        qf_arch='256-256-256',

        n_epochs=200,
        n_env_steps_per_epoch=1000,
        n_train_step_per_epoch=1000,
        eval_period=10,
        eval_n_trajs=5,
        batch_size=256,

        ddpg=dict(DDPG.get_default_config()),
        logging=dict(WandBLogger.get_default_config()),
        env_params=ENV_PARAMS
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="carla-pid")
    parser.add_argument("--n_env_steps_per_epoch", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--eval_period", type=int, default=10)
    parser.add_argument("--policy_arch", type=str, default="256-256")
    parser.add_argument("--qf_arch", type=str, default="256-256")

    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--noise_max_steps", type=int, default=100000)
    parser.add_argument("--reward_scale", type=float, default=1)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--qf_lr", type=float, default=3e-4)
    parser.add_argument("--soft_target_update_rate", type=float, default=5e-3)
    parser.add_argument("--target_update_period", type=int, default=1)
    args = parser.parse_args()

    # update general parameters
    variant["n_env_steps_per_epoch"] = args.n_env_steps_per_epoch
    variant["eval_period"] = args.eval_period
    variant["noise_max_steps"] = args.noise_max_steps
    variant["env"] = args.env
    variant["n_epochs"] = args.n_epochs
    variant["policy_arch"] = args.policy_arch
    variant["qf_arch"] = args.policy_arch           # CHANGE THIS TO QF_ARCH

    # update sac parameters
    variant["ddpg"]["discount"] = args.discount
    variant["ddpg"]["reward_scale"] = args.reward_scale
    variant["ddpg"]["policy_lr"] = args.policy_lr
    variant["ddpg"]["qf_lr"] = args.qf_lr
    variant["ddpg"]["soft_target_update_rate"] = args.soft_target_update_rate
    variant["ddpg"]["target_update_period"] = args.target_update_period
    
    main(variant)
