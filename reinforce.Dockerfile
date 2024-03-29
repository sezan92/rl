FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY = ${DISPLAY}
RUN echo "export DISPLAY=:${DISPLAY}" >> /etc/profile
COPY requirements.txt /requirements.txt
COPY rl /rl
RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-get -y update && \
    apt-get install -y \
    tzdata \
    unzip libglu1-mesa-dev \
    libgl1-mesa-dev libosmesa6-dev \
    xvfb patchelf ffmpeg cmake swig \
    x11-xserver-utils \
    python3-pip python3-tk && \
    pip3 install --upgrade pip && \
    pip3 install -r /requirements.txt
RUN pip3 install /rl 
