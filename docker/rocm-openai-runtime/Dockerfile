ARG ROCM_DEV_IMAGE=rocm/dev-ubuntu-24.04:7.2-complete
FROM ${ROCM_DEV_IMAGE}

SHELL ["/bin/bash", "-lc"]

ARG AMD_LLAMA_PACKAGE_URL=https://repo.radeon.com/rocm/llama.cpp/linux/rocm-rel-7.2/llama-b7782-ubuntu-24.04-rocm-7.2.0-gfx110X-gfx115X-gfx120X-x64.zip
ARG VLLM_PIP_SPEC=vllm

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    unzip \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/amd-llama \
 && curl -fsSL "${AMD_LLAMA_PACKAGE_URL}" -o /tmp/amd-llama.zip \
 && unzip -q /tmp/amd-llama.zip -d /opt/amd-llama \
 && rm -f /tmp/amd-llama.zip

RUN python3 -m pip install --no-cache-dir --break-system-packages ${VLLM_PIP_SPEC}

COPY backend/app/runtime/managed_openai_runtime.py /opt/rocm-openai-runtime/managed_openai_runtime.py
COPY docker/rocm-openai-runtime/entrypoint.sh /usr/local/bin/rocm-openai-entrypoint
RUN chmod +x /usr/local/bin/rocm-openai-entrypoint

WORKDIR /work
ENTRYPOINT ["rocm-openai-entrypoint"]
