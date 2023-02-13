FROM ubuntu:focal
MAINTAINER Simon Perkins "simon.perkins@gmail.com"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update && \
    apt -y upgrade && \
    apt install -y \
        git \
        python3-pip \
        python3-minimal \
        python-is-python3 \
        wget \
        rsync && \
    apt clean

ADD . /src/predict

RUN python -m pip install -U pip setuptools wheel && \
    python -m pip install /src/predict && \
    python -m pip cache purge
