# syntax = devthefuture/dockerfile-x
# SPDX-License-Identifier: MIT

# Install dependencies
RUN apt-get update && \
        apt-get install --no-install-recommends -y \
            build-essential \
            zlib1g-dev \
            libx11-dev \
            libusb-1.0-0-dev \
            freeglut3-dev \
            liblapacke-dev \
            libopenblas-dev \
            libatlas-base-dev \
            cmake \
            git-core \
            libgtk-3-dev \
            pkg-config \
            clang && \
        apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Clone the repo and install the python bindings
ARG LIBSURVIVE_DIR=${USERHOME}/libsurvive
RUN git clone https://github.com/cntools/libsurvive.git --recursive ${LIBSURVIVE_DIR} && \
        cd ${LIBSURVIVE_DIR} && \
        CC=clang CXX=clang++ pip install . --no-deps --verbose
