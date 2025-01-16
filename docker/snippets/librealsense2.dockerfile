# syntax = devthefuture/dockerfile-x
# Install librealsense2

# Install prerequisites
RUN apt-get update && apt-get install -y \
    libssl-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    pkg-config \
    libgtk-3-dev \
    git-core \
    wget \
    cmake \
    build-essential \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    at \
    v4l-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-gst-1.0 \
    gir1.2-gst-rtsp-server-1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-ugly \
    libx264-dev \
    python3-opencv \
    libxinerama-dev \
    libsdl2-dev \
    curl \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the librealsense repo
ARG LIBREALSENSE_DIR="${USERHOME}/librealsense"
RUN git clone https://github.com/IntelRealSense/librealsense.git ${LIBREALSENSE_DIR}

# Build and install librealsense
RUN cd ${LIBREALSENSE_DIR} \
    && mkdir build && cd build \
    && cmake \
        -DBUILD_SHARED_LIBS=false \
        -DBUILD_PYTHON_BINDINGS=true \
        -DPYTHON_EXECUTABLE=$(which python) \
        -DCMAKE_BUILD_TYPE=Release \
        -DOpenGL_GL_PREFERENCE=GLVND \
        -DBUILD_EXAMPLES=false \
        -DBUILD_GRAPHICAL_EXAMPLES=false \
        -DFORCE_RSUSB_BACKEND=ON .. \
    && make -j$(nproc) \
    && make install

# Copy the realsense config to udev rules
RUN mkdir -p /etc/udev/rules.d/ && \
        cp ${LIBREALSENSE_DIR}/config/99-realsense-libusb.rules /etc/udev/rules.d/

ENV PYTHONPATH=$PYTHONPATH:${LIBREALSENSE_DIR}/build/Release
