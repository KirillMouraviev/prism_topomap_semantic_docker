# syntax=docker/dockerfile:1

###############################################################################
### 1. Builder stage
###############################################################################
ARG CUDA_TAG=12.1.1-cudnn8-devel-ubuntu20.04

FROM nvcr.io/nvidia/cuda:${CUDA_TAG} AS builder

ENV DEBIAN_FRONTEND=noninteractive

ARG MAX_JOBS=4

ARG PYTORCH_CUDA=cu121
ARG PYTORCH_VERSION=2.1.2
ARG TORCHVISION_VERSION=0.16.2
ARG NUMPY_VERSION=1.24.4
ARG ME_COMMIT=4b628a7
ARG FAISS_COMMIT=e45ae24

ARG PIP_VERSION=25.0.1
ARG WHEEL_VERSION=0.45.1
ARG SETUPTOOLS_VERSION=69.0.3
ARG NINJA_VERSION=1.11.1.1

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cmake \
        wget \
        swig \
        ninja-build \
        python3-dev \
        python3-pip \
        libopenblas-dev \
        libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# --Python base---------------------------------------------------------------
RUN python3 -m pip --no-cache-dir install \
        pip==${PIP_VERSION} \
        wheel==${WHEEL_VERSION} \
        setuptools==${SETUPTOOLS_VERSION} \
        ninja==${NINJA_VERSION} \
        numpy==${NUMPY_VERSION}

###############################################################################
### 1a. PyTorch wheel (including torchvision and dependencies)
###############################################################################
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels \
        torch==${PYTORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}
# MinkowskiEngine require torch installation to build itself
RUN python3 -m pip install --no-cache-dir /wheels/torch*.whl

###############################################################################
### 1b. MinkowskiEngine wheel
###############################################################################
WORKDIR /build/mink
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUDA_HOME=/usr/local/cuda-12.1
RUN git clone --recursive https://github.com/alexmelekhin/MinkowskiEngine.git \
        && cd MinkowskiEngine \
        && git checkout 6532dc3 \
        && python3 setup.py bdist_wheel \
                --force_cuda \
                --blas=openblas \
                --dist-dir /wheels

###############################################################################
### 1c. Faiss-GPU wheel
###############################################################################
# upgrade cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.5/cmake-3.26.5-linux-x86_64.sh && \
    mkdir /opt/cmake-3.26.5 && \
    bash cmake-3.26.5-linux-x86_64.sh --skip-license --prefix=/opt/cmake-3.26.5/ && \
    ln -s /opt/cmake-3.26.5/bin/* /usr/local/bin && \
    rm cmake-3.26.5-linux-x86_64.sh
WORKDIR /build/faiss
RUN git clone https://github.com/facebookresearch/faiss.git \
    && cd faiss \
    && git checkout c3b93749 \
    && cmake -B build . \
        -Wno-dev \
        -DFAISS_ENABLE_GPU=ON \
        -DFAISS_ENABLE_PYTHON=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDAToolkit_ROOT=${CUDA_HOME} \
        -DCMAKE_CUDA_ARCHITECTURES="60;61;70;75;80;86" \
    && make -C build -j${MAX_JOBS} faiss \
    && make -C build -j${MAX_JOBS} swigfaiss \
    && cd build/faiss/python \
    && python3 setup.py bdist_wheel --dist-dir /wheels

###############################################################################
### 2. Dev/runtime stage
###############################################################################
FROM nvcr.io/nvidia/cuda:${CUDA_TAG} AS dev

ENV DEBIAN_FRONTEND=noninteractive

ARG INSTALL_ROS1=false
ENV ROS_DISTRO=noetic

# — lightweight system packages for interactive work —
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        python-is-python3 \
        git \
        nano \
        vim \
        sudo \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

RUN apt-get update && apt-get install -y libopengl-dev libgl1-mesa-dev python3-opengl xorg-dev libglu1-mesa-dev
ENV CMAKE_POLICY_VERSION_MINIMUM 3.5
RUN git clone --branch v0.2.3 https://github.com/facebookresearch/habitat-sim \
    && cd habitat-sim && python3 -m pip install --no-cache-dir -e . \
    && python setup.py install

# — copy the compiled wheels and install them —
COPY --from=builder /wheels /tmp/wheels
RUN rm -rf /tmp/wheels/pillow* \
    && python3 -m pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# WARNING: This allows sudo without password for all users in the sudo group
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# install some needed packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y kmod kbd tmux
RUN apt-get update \
    && apt-get install -y libopenblas-dev ffmpeg libsm6 libxext6

# - optional ROS1 Noetic installation -
RUN if [ "$INSTALL_ROS1" = "true" ]; then \
    apt-get update \
    && apt-get install -y lsb-release \
    && apt-get clean all; \
    fi
RUN if [ "$INSTALL_ROS1" = "true" ]; then \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - \
    && apt -y update \
    && apt install -y ros-${ROS_DISTRO}-desktop-full \
    && apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential; \
    fi

# Install challenge specific habitat-lab
RUN pip install --upgrade setuptools \
    && pip install --upgrade pip
RUN git clone --branch toposlam_experiments https://github.com/kirillmouraviev/habitat-lab
RUN cd habitat-lab \
    && python3 -m pip install --no-cache-dir ./habitat-lab
RUN cd habitat-lab \
    && python3 -m pip install --no-cache-dir ./habitat-baselines

# Install needed python packages
RUN python3 -m pip install rosnumpy keyboard scikit-image

# — create a user with the host's UID and GID —
ARG USER_NAME=docker_prism
ARG HOST_UID=1000
ARG HOST_GID=1000
ENV HOME=/home/${USER_NAME}
RUN groupadd --gid ${HOST_GID} ${USER_NAME} \
    && useradd --uid ${HOST_UID} \
               --gid ${HOST_GID} \
               --create-home \
               --shell /bin/bash \
               ${USER_NAME} \
    && usermod -aG sudo ${USER_NAME}

# — working directory for convenience —
WORKDIR ${HOME}
USER ${USER_NAME}

# Install Open3D (needed for OpenPlaceRecognition)
RUN python3 -m pip install open3d

# Install OpenPlaceRecognition
RUN cd ${HOME} \
    && git clone --branch feat/toposlam https://github.com/OPR-Project/OpenPlaceRecognition \
    && cd OpenPlaceRecognition \
    && python3 -m pip install -e .
COPY minkloc3d_nclt.pth ${HOME}/OpenPlaceRecognition/weights/place_recognition/

RUN pip install protobuf==3.20.0 && \
    pip install loguru && \
    pip install memory_profiler

RUN pip install setuptools==66.0.0 && \
    pip install netifaces && \
    pip install pycryptodomex && \
    pip install git+https://github.com/ros/genpy.git && \
    pip install python-gnupg && \
    pip install transformations

EXPOSE 8888

EXPOSE 6006

COPY image /
#COPY habitat-challenge-data /data_config
ENV SHELL /bin/bash

ENV JUPYTER_PASSWORD "jupyter"
ENV JUPYTER_TOKEN "jupyter"

RUN sudo chmod 777 /startup.sh
RUN sudo chmod 777 /usr/local/bin/jupyter.sh
RUN sudo chmod 777 /usr/local/bin/xvfb.sh

ENTRYPOINT ["/startup.sh"]
CMD ["/bin/bash"]
