FROM nvidia/cuda:11.1-base-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
COPY requirements.txt /requirements.txt
COPY rl /rl
COPY setup.py /rl/setup.py
RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get -y update && \
    apt-get install -y \
    tzdata \
    unzip libglu1-mesa-dev \
    libgl1-mesa-dev libosmesa6-dev \
    xvfb patchelf ffmpeg cmake swig \
    python3-pip && \
    pip3 install -r requirements.txt
RUN python3 -m pip install -e /rl 