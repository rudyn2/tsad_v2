FROM carlasim/carla:0.9.11

# SYSTEM LEVEL DEPENDENCIES
USER root
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libjpeg-turbo8 -y
RUN apt-get install -y xdg-user-dirs xdg-utils
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# ADD CUSTOM USER
ARG user_name=client
ARG PATH="/home/$user_name/miniconda3/bin:${PATH}"
RUN useradd -ms /bin/bash $user_name
ENV PATH="/home/$user_name/miniconda3/bin:${PATH}"
USER $user_name
WORKDIR /home/$user_name

# INSTALL CONDA
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN mkdir /home/$user_name/.conda
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm -f Miniconda3-latest-Linux-x86_64.sh

# CREATE CONDA ENV AND INSTALL PYTHON DEPENDENCIES
RUN conda create -n carla_env python=3.7
ENV PATH /home/$user_name/miniconda3/envs/carla_env/bin:$PATH
ENV CONDA_DEFAULT_ENV carla_env
RUN conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
RUN conda install numpy

# COPY WHEELS AND INSTALL
COPY wheels/gym_carla-0.1.0-py3-none-any.whl gym_carla-0.1.0-py3-none-any.whl
RUN pip install gym_carla-0.1.0-py3-none-any.whl

RUN mkdir -p /home/$user_name/workspace
ENV PYTHONPATH "${PYTHONPATH}:/home/$user_name/workspace"
ENV PYTHONPATH "${PYTHONPATH}:/home/$user_name/workspace/carla_egg/carla-0.9.11-py3.7-linux-x86_64.egg"

COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt

ENV CARLA_PORT 2000
ENV CARLA_TM_PORT 8000
ENV CARLA_HOST="localhost"

WORKDIR /home/$user_name/workspace
# RUN nohup /bin/bash -c "SDL_VIDEODRIVER=offscreen /home/carla/CarlaUE4.sh" &>/dev/null &
# RUN nohup /bin/bash -c "SDL_VIDEODRIVER=offscreen /home/carla/CarlaUE4.sh -quality-level=low -world-port=2000 -benchmark -carla-server -carla-no-hud" &>/dev/null &
# COPY . .
