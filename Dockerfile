############################
# Shared build arguments
############################
ARG CUDA_BASE=nvidia/cuda:13.1.1-runtime-ubuntu24.04
ARG PYTHON_VERSION=3.12

############################
# Base image with all setup
############################
FROM ${CUDA_BASE} AS base

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV UV_INSTALL_DIR=/opt/uv
ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_HOME=/opt/cargo
ENV PATH="/opt/uv:/root/.local/bin:/opt/cargo/bin:${PATH}"

RUN apt-get update -o Acquire::Retries=5 && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    pkg-config \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh && \
    sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain stable && \
    rm -f /tmp/rustup-init.sh

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv --version

WORKDIR /workspace

# Copy package metadata and source needed by editable/path dependency installs.
COPY pyproject.toml uv.lock README.md /workspace/
COPY src /workspace/src
COPY regex_dfa_guide/pyproject.toml regex_dfa_guide/Cargo.toml regex_dfa_guide/Cargo.lock regex_dfa_guide/LICENSE /workspace/regex_dfa_guide/
COPY regex_dfa_guide/python /workspace/regex_dfa_guide/python
COPY regex_dfa_guide/src /workspace/regex_dfa_guide/src

# Install Python and project dependencies
RUN uv python install ${PYTHON_VERSION}
RUN uv sync --python ${PYTHON_VERSION} --group dev

# Copy the full source tree later
COPY . /workspace

# Build optional native evaluation artifact
RUN .venv/bin/python scripts/build_wd_kernel.py

CMD ["/bin/bash"]

############################
# Final GPU + coverage target
############################
FROM base AS latest-cov

RUN rustup component add llvm-tools-preview --toolchain stable && \
    cargo install cargo-llvm-cov

############################
# Final GPU target (default)
############################
FROM base AS latest
