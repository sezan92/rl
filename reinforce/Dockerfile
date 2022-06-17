ARG DEBIAN_FRONTEND=noninteractive
FROM nvidia/cuda:11.1-base-ubuntu20.04
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get -y update && \
    apt-get install -y \
    unzip libglu1-mesa-dev \
    libgl1-mesa-dev libosmesa6-dev \
    xvfb patchelf ffmpeg cmake swig \
    python3-pip
RUN pip3 install requirements.txt