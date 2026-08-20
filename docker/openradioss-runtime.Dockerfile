FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgcc-s1 \
        libgfortran5 \
        libgomp1 \
        libstdc++6 \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

