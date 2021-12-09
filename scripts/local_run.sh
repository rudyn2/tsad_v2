#!/bin/bash

GPU=0
NAME=tsad2
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
echo "Connection to network: ${NETWORK}"

docker run --name ${NAME} --rm --runtime=nvidia --network ${NETWORK} -e NVIDIA_VISIBLE_DEVICES=${GPU} -it -v $(pwd):/home/workspace tsad2 /bin/bash