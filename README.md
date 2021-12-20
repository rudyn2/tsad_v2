# tsad_v2
Autonomous Driving with Deep Reinforcement Learning

    docker run --name tsad2 --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -it -v $(pwd):/home/client/workspace tsad2 /bin/bash

    nohup /bin/bash -c "SDL_VIDEODRIVER=offscreen /home/carla/CarlaUE4.sh -quality-level=low -world-port=2000 -benchmark -carla-server -carla-no-hud" &>/dev/null &