FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update and install essential tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    curl \
    pkg-config \
    libopencv-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# 1. Install cuDNN 9 for CUDA 12
RUN wget https://developer.download.nvidia.com/compute/cudnn/9.1.1/local_installers/cudnn-local-repo-ubuntu2204-9.1.1_1.0-1_amd64.deb && \
    dpkg -i cudnn-local-repo-ubuntu2204-9.1.1_1.0-1_amd64.deb && \
    cp /var/cudnn-local-repo-ubuntu2204-9.1.1/cudnn-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install libcudnn9-cuda-12 && \
    rm cudnn-local-repo-ubuntu2204-9.1.1_1.0-1_amd64.deb

# 2. Download and install ONNX Runtime GPU (v1.25.0)
ENV ORT_VERSION=1.25.0
ENV ORT_DIR=/opt/onnxruntime
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz -O ort.tgz && \
    tar -xzf ort.tgz && \
    mv onnxruntime-linux-x64-gpu-${ORT_VERSION} ${ORT_DIR} && \
    rm ort.tgz

# 3. Install GStreamer build dependencies and plugins
RUN apt-get update && apt-get install -y \
    meson \
    ninja-build \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    flex \
    bison \
    && rm -rf /var/lib/apt/lists/*

# Set Environment Variables
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${ORT_DIR}/lib:${LD_LIBRARY_PATH}
ENV ORT_ROOT=${ORT_DIR}

WORKDIR /workspace
VOLUME ["/workspace"]

CMD ["/bin/bash"]
