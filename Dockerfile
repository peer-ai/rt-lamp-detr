FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*

RUN pip --no-cache-dir install Cython

COPY requirements.txt /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt

WORKDIR /workspace
