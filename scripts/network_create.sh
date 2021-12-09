#!/bin/bash

NAME=tsad2

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -n|--name)
      NAME="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

echo "Bridge name: ${NAME}"

docker network create --driver bridge ${NAME}
docker network inspect ${NAME}