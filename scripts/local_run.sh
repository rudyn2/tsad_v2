#!/bin/bash

docker run --name tsad2 --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -it -v $(pwd):/home/workspace tsad2 /bin/bash