#!/bin/bash

GPU=0
PORT=2000
NAME=carla-server
NETWORK=tsad2

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -g|--gpu)
      GPU="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--port)
      PORT="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--name)
      NAME="$2"
      shift # past argument
      shift # past value
      ;;
    -N|--network)
      NETWORK="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

echo "Using GPU number: ${GPU}"
echo "Connection to CARLA through port : ${PORT}"
echo "Connection to network: ${NETWORK}"

docker run --name ${NAME} -d --network ${NETWORK} --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${GPU} -it carlasim/carla:0.9.11 /bin/bash -c "SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl -quality-level=low -world-port=${PORT} -benchmark -carla-server -carla-no-hud"