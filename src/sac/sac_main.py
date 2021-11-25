from copy import deepcopy
import argparse
import numpy as np
from gym_carla.envs.carla_env import CarlaEnv
from gym_carla.envs.carla_pid_env import CarlaPidEnv

from .sac import SAC
from ..models.replay_buffer import ReplayBuffer, batch_to_torch
from ..models.model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from ..utils.sampler import StepSampler, TrajSampler
from ..utils.utils import Timer, set_random_seed, prefix_metrics
from ..utils.utils import WandBLogger


ENV_PARAMS = {
            # carla connection parameters+
            'host': 'localhost',
            'port': '2000',  # connection port
            'town': 'Town01',  # which town to simulate
            'traffic_manager_port': 8000,

            # simulation parameters
            'verbose': False,
            'vehicles': 100,  # number of vehicles in the simulation
            'walkers': 10,  # number of walkers in the simulation
            'obs_size': 224,  # sensor width and height
            'max_past_step': 1,  # the number of past steps to draw
            'dt': 0.025,  # time interval between two frames
            # reward weights [speed, collision, lane distance]
            'reward_weights': [0.3, 0.3, 0.3],
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


def main(variant):
    wandb_logger = WandBLogger(config=WandBLogger.get_default_config(), variant=variant)
    wandb_config = wandb_logger.wandb_config
    set_random_seed(wandb_config["seed"].seed)

    # integrate gym-carla here
    env_params = wandb_config["env_params"]
    if args.act_mode == "pid":
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

    train_sampler = StepSampler(carla_env, wandb_config["max_traj_length"])
    eval_sampler = TrajSampler(carla_env, wandb_config["max_traj_length"])

    replay_buffer = ReplayBuffer(wandb_config["replay_buffer_size"])

    policy = TanhGaussianPolicy(
        train_sampler.env.observation_space.shape[0],
        train_sampler.env.action_space.shape[0],
        wandb_config["policy_arch"],
        log_std_multiplier=wandb_config["policy_log_std_multiplier"],
        log_std_offset=wandb_config["policy_log_std_offset"],
    )

    qf1 = FullyConnectedQFunction(
        train_sampler.env.observation_space.shape[0],
        train_sampler.env.action_space.shape[0],
        wandb_config["qf_arch"]
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        train_sampler.env.observation_space.shape[0],
        train_sampler.env.action_space.shape[0],
        wandb_config["qf_arch"]
    )
    target_qf2 = deepcopy(qf2)

    if wandb_config["trainer_kwargs"]["target_entropy"] >= 0.0:
        wandb_config["trainer_kwargs"]["target_entropy"] = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = SAC(wandb_config["sac_config"], policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(wandb_config["device"])

    sampler_policy = SamplerPolicy(policy, wandb_config["device"])

    for epoch in range(wandb_config['n_epochs']):
        metrics = {}
        with Timer() as rollout_timer:
            train_sampler.sample(
                sampler_policy, wandb_config["n_env_steps_per_epoch"],
                deterministic=False, replay_buffer=replay_buffer
            )
            metrics['env_steps'] = replay_buffer.total_steps
            metrics['epoch'] = epoch

        with Timer() as train_timer:
            for batch_idx in range(wandb_config["n_train_step_per_epoch"]):
                batch = batch_to_torch(replay_buffer.sample(wandb_config["batch_size"]), wandb_config["device"])
                if batch_idx + 1 == wandb_config["n_train_step_per_epoch"]:
                    metrics.update(prefix_metrics(sac.train(batch), 'sac'))
                else:
                    sac.train(batch)

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % wandb_config["eval_period"] == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, wandb_config["eval_n_trajs"], deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])

        metrics['rollout_time'] = rollout_timer()
        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = rollout_timer() + train_timer() + eval_timer()
        wandb_logger.log(metrics)


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        max_traj_length=1000,
        replay_buffer_size=1000000,
        seed=42,
        device='cuda',

        policy_arch='256-256',
        qf_arch='256-256',
        policy_log_std_multiplier=1.0,
        policy_log_std_offset=-1.0,

        n_epochs=2000,
        n_env_steps_per_epoch=1000,
        n_train_step_per_epoch=1000,
        eval_period=10,
        eval_n_trajs=5,

        batch_size=256,

        sac=SAC.get_default_config(),
        logging=WandBLogger.get_default_config(),
        env_params=ENV_PARAMS
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str, help="Data path.")
    parser.add_argument("--checkpoint_path", required=True, default="", type=str, help="Checkpoint path.")

    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--layer_size", default=128, type=int, help="Layer size")

    parser.add_argument("--epochs", default=1000, type=int, help="Epochs.")
    parser.add_argument("--num_trains_per_train_loop", default=2, type=int, help="Num batch updates per epoch.")
    parser.add_argument("--gpu", default='0', type=str)
    # if we want to try max_{a'} backups, set this to true
    parser.add_argument("--max_q_backup", type=str, default="False")
    # defaults to true, it does not backup entropy in the Q-function, as per Equation 3
    parser.add_argument("--deterministic_backup", type=str, default="True")
    # Defaulted to 20000 (40000 or 10000 work similarly)
    parser.add_argument("--policy_eval_start", default=0, type=int)
    # the value of alpha, set to 5.0 or 10.0 if not using lagrange
    parser.add_argument('--min_q_weight', default=1.0, type=float)
    parser.add_argument('--policy_lr', default=1e-4, type=float)  # Policy learning rate
    parser.add_argument('--qf_lr', default=1e-4, type=float)  # Policy learning rate
    parser.add_argument('--min_q_version', default=3, type=int)  # min_q_version = 3 (CQL(H)), version = 2 (CQL(rho))
    parser.add_argument('--reward_scale', default=1.0, type=float)
    parser.add_argument('--data_percentage', default=1.0, type=float)

    parser.add_argument("--wandb", default=True, type=bool, help="Wheter to log in wandb or not")
    parser.add_argument("--progress-bar", action="store_true", help="Wheter to use progress bar or not")
    parser.add_argument('--seed', default=10, type=int)

    args = parser.parse_args()

    # TRAINER KWARGS
    variant['trainer_kwargs']['max_q_backup'] = (True if args.max_q_backup == 'True' else False)
    variant['trainer_kwargs']['deterministic_backup'] = (True if args.deterministic_backup == 'True' else False)
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['qf_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['reward_scale'] = args.reward_scale
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh
    if args.lagrange_thresh <= 0.0:
        variant['trainer_kwargs']['with_lagrange'] = False

    # ALGORITHM KWARGS
    variant["algorithm_kwargs"]["progress_bar"] = args.progress_bar
    variant["algorithm_kwargs"]["log_wandb"] = args.wandb
    variant["algorithm_kwargs"]["batch_size"] = args.batch_size
    variant["algorithm_kwargs"]["num_epochs"] = args.epochs
    variant["algorithm_kwargs"]["num_trains_per_train_loop"] = args.num_trains_per_train_loop
    if args.checkpoint_path != "":
        variant["algorithm_kwargs"]["checkpoint_metric"] = "dataset_q1_values"
        variant["algorithm_kwargs"]["save_checkpoint"] = True
        variant["algorithm_kwargs"]["checkpoint_path"] = args.checkpoint_path

    # GENERAL ARGS
    variant["offline_buffer"] = args.data
    variant['seed'] = args.seed
    variant['layer_size'] = args.layer_size
    variant['data_percentage'] = args.data_percentage

    rnd = np.random.randint(low=0, high=1000000)
    main(variant)
