ARG ROCM_DEV_IMAGE=rocm/dev-ubuntu-24.04:7.2-complete
FROM ${ROCM_DEV_IMAGE}

SHELL ["/bin/bash", "-lc"]

ARG AMD_LLAMA_PACKAGE_URL=https://repo.radeon.com/rocm/llama.cpp/linux/rocm-rel-7.2/llama-b7782-ubuntu-24.04-rocm-7.2.0-gfx110X-gfx115X-gfx120X-x64.zip
ARG INSTALL_VLLM=0
ARG VLLM_INSTALL_METHOD=wheel
ARG VLLM_WHEEL_EXTRA_INDEX_URL=https://wheels.vllm.ai/rocm/
ARG ROCM_WHL_URL=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
ARG ROCM_TORCH_VERSION=2.10.0+rocm7.2.0.lw.gitb6ee5fde
ARG ROCM_TORCHAUDIO_VERSION=2.10.0+rocm7.2.0.git5047768f
ARG ROCM_TORCHVISION_VERSION=0.25.0+rocm7.2.0.git82df5f59
ARG ROCM_NIGHTLY=0
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=v0.17.0

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
 && find /opt/amd-llama -type f -name 'llama-*' -exec chmod +x {} + \
 && rm -f /tmp/amd-llama.zip

RUN if [ "${INSTALL_VLLM}" = "1" ]; then \
      if [ "${VLLM_INSTALL_METHOD}" = "wheel" ]; then \
        python3 -m pip install --no-cache-dir --break-system-packages \
          --extra-index-url "${VLLM_WHEEL_EXTRA_INDEX_URL}" \
          vllm; \
      else \
        if [ "${ROCM_NIGHTLY}" = "1" ]; then \
          python3 -m pip install --no-cache-dir --break-system-packages --pre --index-url "${ROCM_WHL_URL}" \
            torch torchvision torchaudio; \
        else \
          python3 -m pip install --no-cache-dir --break-system-packages \
            --find-links "${ROCM_WHL_URL}" \
            "torch==${ROCM_TORCH_VERSION}" \
            "torchaudio==${ROCM_TORCHAUDIO_VERSION}" \
            "torchvision==${ROCM_TORCHVISION_VERSION}"; \
        fi; \
        git clone --depth 1 --branch "${VLLM_REF}" "${VLLM_REPO}" /tmp/vllm; \
        cd /tmp/vllm; \
        python3 use_existing_torch.py; \
        python3 -m pip install --no-cache-dir --break-system-packages -r requirements/rocm.txt; \
        python3 -m pip install --no-cache-dir --break-system-packages --no-build-isolation .; \
        rm -rf /tmp/vllm; \
      fi; \
    fi

COPY backend/app/runtime/managed_openai_runtime.py /opt/rocm-openai-runtime/managed_openai_runtime.py
COPY docker/rocm-openai-runtime/entrypoint.sh /usr/local/bin/rocm-openai-entrypoint
RUN chmod +x /usr/local/bin/rocm-openai-entrypoint

WORKDIR /work
ENTRYPOINT ["rocm-openai-entrypoint"]
