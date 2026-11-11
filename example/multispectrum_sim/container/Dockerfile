ARG CUDA_BASE_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build libeigen3-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENTRYPOINT []
CMD ["/bin/bash"]
