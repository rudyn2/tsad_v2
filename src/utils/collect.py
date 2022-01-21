import random
import uuid

import numpy as np
from termcolor import colored
import time
import sys
import json
import os
import argparse
from datetime import datetime

import carla
from gym_carla.envs.carla_env import CarlaEnv
from lbc_agent.noisy_agent import NoisyAgent


WEATHER_MORNING = [20.0, 90.0, 30.0, 30.0, 0.0, 30.0]
WEATHER_MIDDAY = [30.0, 0.0, 60.0, 30.0, 0.0, 80.0]
WEATHER_AFTERNOON = [50.0, 0.0, 40.0, 30.0, 0.0, -40.0]
WEATHER_DEFAULT = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
WEATHER_ALMOST_NIGHT = [30.0, 30.0, 0.0, 30.0, 0.0, -60.0]
WEATHER_OPTIONS = [WEATHER_DEFAULT]


class JsonSaver:
    def __init__(self, path: str):
        self.path = path

    def write(self, d: dict):
        if not os.path.exists(self.path):
            with open(self.path, "w+") as file:
                json.dump(d, file)
                return

        with open(self.path, "r+") as file:
            data = json.load(file)
            data.update(d)
            file.seek(0)
            json.dump(data, file)

    def save_one_ego_run(self, info_data: list, run_id: str):

        tries = 3
        while tries > 0:
            try:
                # ego info
                ego_run_info = {run_id: {}}
                for item in info_data:
                    ego_run_info[run_id][item['timestamp']] = item['metadata']
                self.write(ego_run_info)
                break
            except json.JSONDecodeError:
                pass
            tries -= 1


def parse_control(c):
    """
    Parse a carla.VehicleControl to a json object.
    """
    return {
        "brake": c.brake,
        "gear": c.gear,
        "hand_brake": c.hand_brake,
        "manual_gear_shift": c.manual_gear_shift,
        "reverse": c.reverse,
        "steer": c.steer,
        "throttle": c.throttle
    }


def try_set_route(agent: NoisyAgent, agent_location, points: list):
    # find the first location that isn't the ego vehicle location and its far away
    points_distance = []
    for point in points:
        dx = point.location.x - agent_location.x
        dy = point.location.y - agent_location.y
        distance = np.sqrt(dx * dx + dy * dy)
        if point.location != agent_location:
            points_distance.append((point, distance))

    # sort point in descending order according to their distance to the ego
    points_distance = sorted(points_distance, key=lambda x: x[1], reverse=True)
    route_ready = False
    for point, _ in points_distance:
        try:
            agent.set_route(agent_location, point.location)
            route_ready = True
            break
        except Exception:
            continue
    if not route_ready:
        raise RuntimeError("Couldn't set the route. Please try again!")


def main(args):
    # parameters for the gym_carla environment
    params = {
        # carla connection parameters+
        'host': args.host,
        'port': args.port,  # connection port
        'town': 'Town01',  # which town to simulate

        # simulation parameters
        'verbose': False,
        'weaken_terminals': True,   # ADDED TO USE THE LBC AGENT PLANNER
        'vehicles': args.vehicles,  # number of vehicles in the simulation
        'walkers': args.walkers,  # number of walkers in the simulation
        'obs_size': 288,  # sensor width and height
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.025,  # time interval between two frames
        'continuous_throttle_range': [0.0, 1.0],  # continuous acceleration range
        'continuous_brake_range': [0.0, 1.0],  # continuous throttle range
        'continuous_steer_range': [-1.0, 1.0],  # continuous steering angle range
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 6,  # desired speed (m/s)
        'speed_reduction_at_intersection': 0.75,
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    }

    env = CarlaEnv(params)
    info = env.reset()[-1]
    print(colored("Agent ready", "green"))
    output_file = datetime.now().strftime('%Y-%m-%d-%H-%M')
    json_saver = JsonSaver(f"{output_file}.json")
    verbose = True
    total_episodes = args.n
    max_frames = args.T

    try:
        for episode in range(total_episodes):
            env.world.set_weather(carla.WeatherParameters(*random.choice(WEATHER_OPTIONS)))
            ego_vehicle = env.ego
            # ego_vehicle_location = ego_vehicle.get_location()
            ego_agent = NoisyAgent(ego_vehicle, is_noisy=args.noise)
            # try_set_route(ego_agent, ego_vehicle_location, env.map.get_spawn_points())
            episode_id = str(uuid.uuid4())
            episode_affordances = {}
            episode_metadata = []
            frames_count = 0

            done = False
            while not done:
                start = time.time()
                timestamp = round(start * 1000.0)
                target_transform = carla.Transform(carla.Location(x=float(info['waypoints'][2][0]),
                                                                  y=float(info['waypoints'][2][1]),
                                                                  z=0.0))
                hlc_plus_1 = int(info['road_option'].value)
                ego_info = ego_agent.run_step_with_planner(target_transform, hlc_plus_1)
                control = ego_info["control"]
                action = [control.throttle, control.brake, control.steer]

                obs, r, done, info = env.step(action)
                rl_info_ = info.copy()
                rl_info_['road_option'] = rl_info_['road_option'].value
                ego_info['control'] = parse_control(control)
                rl_info = dict(obs=obs["affordances"].tolist(),
                               action=action,
                               reward=r,
                               done=done,
                               info=rl_info_,
                               hlc=hlc_plus_1 - 1)
                ego_info['rl'] = rl_info
                frames_count += 1

                # skip initial frames
                if frames_count >= args.skip_frames:
                    episode_affordances[str(timestamp)] = obs["affordances"]
                    episode_metadata.append(dict(timestamp=timestamp, metadata=ego_info))

                if verbose:
                    fps = 1 / (time.time() - start)
                    sys.stdout.write("\r")
                    sys.stdout.write(f"[{fps:.1f} fps] HLC={hlc_plus_1 - 1} "
                                     f"traffic_light_state={obs['affordances'][0]:.3f} "
                                     f"distance_to_traffic_light={obs['affordances'][1]:.3f} "
                                     f"is_vehicle_hazard={obs['affordances'][2]:.3f} "
                                     f"front_vehicle_distance={obs['affordances'][3]:.3f} \n"
                                     f"lane_dis={obs['affordances'][7]:.3f} "
                                     f"delta_yaw={obs['affordances'][8]:.3f} "
                                     f"ego_v_x={obs['affordances'][9]:.3f} "
                                     f"ego_v_y={obs['affordances'][10]:.3f} "
                                     f"ego_v_z={obs['affordances'][11]:.3f} "
                                     f"ego_steer={obs['affordances'][12]:.3f} \n"
                                     f"last_ego_steer={obs['affordances'][13]:.3f} "
                                     f"avg yaw={obs['affordances'][14]:.3f} "
                                     f"({frames_count}/{max_frames}) rew={r:.2f}")
                    sys.stdout.flush()
                done = done or (frames_count >= max_frames)
                if done:
                    env.reset()
                    break

            if verbose:
                print(f"\nEpisode {episode} ({episode_id}) ready with {len(episode_metadata)} frames")

            saving_start = time.time()
            json_saver.save_one_ego_run(episode_metadata, run_id=episode_id)
            np.savez(f"{output_file}_{episode_id}.npz", **episode_affordances)

            if verbose:
                print(f"Saved in {(time.time() - saving_start):.0} seconds")
    finally:
        print(colored("Script finished! Closing...", "green"))
        env._destroy_actors()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Settings for the data capture",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-H', '--host', default='localhost', type=str, help='CARLA server ip address')
    parser.add_argument('-p', '--port', default=2000, type=int, help='CARLA server port number')
    parser.add_argument('-n', default=2, type=int, help='number of ego executions')
    parser.add_argument('-T', default=200, type=int,
                        help='number of frames to record per ego execution')
    parser.add_argument('-t', '--town', default='Town01', type=str, help="town to use")
    parser.add_argument('-ve', '--vehicles', default=100, type=int,
                        help="number of vehicles to spawn in the simulation")
    parser.add_argument('-wa', '--walkers', default=50, type=int, help="number of walkers to spawn in the simulation")
    parser.add_argument('--skip-frames', default=20, type=int)
    parser.add_argument('--noise', action="store_true")
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    main(args)
