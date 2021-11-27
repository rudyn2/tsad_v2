# tsad_v2
Autonomous Driving with Deep Reinforcement Learning

## CARLA

    docker run --name carla-server -d --rm -p 2000-2002:2000-2002 --runtime=nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it carlasim/carla:0.9.11 /bin/bash -c './CarlaUE4.sh -opengl -quality-level=low -world-port=2000'