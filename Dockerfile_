FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
USER root

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libjpeg-turbo8 -y

# Copy wheels and install
COPY wheels/gym_carla-0.1.0-py3-none-any.whl gym_carla-0.1.0-py3-none-any.whl
RUN pip install gym_carla-0.1.0-py3-none-any.whl

# Set some envirnonment variables
ENV PYTHONPATH "${PYTHONPATH}:/home/workspace"
ENV PYTHONPATH "${PYTHONPATH}:/home/workspace/carla_egg/carla-0.9.11-py3.7-linux-x86_64.egg"

COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt

WORKDIR /home/workspace
# COPY . .