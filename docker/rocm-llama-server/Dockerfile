# syntax=docker/dockerfile:1

ARG ROCM_DEV_IMAGE=rocm/dev-ubuntu-24.04:7.2-complete
FROM ${ROCM_DEV_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG AMD_LLAMA_PACKAGE_URL=https://repo.radeon.com/rocm/llama.cpp/linux/rocm-rel-7.2/llama-b7782-ubuntu-24.04-rocm-7.2.0-gfx110X-gfx115X-gfx120X-x64.zip

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/amd-llama \
    && curl -fsSL "${AMD_LLAMA_PACKAGE_URL}" -o /tmp/amd-llama.zip \
    && unzip -q /tmp/amd-llama.zip -d /opt/amd-llama \
    && rm -f /tmp/amd-llama.zip \
    && find /opt/amd-llama -type f -name 'llama-*' -exec chmod +x {} +

COPY docker/rocm-llama-server/entrypoint.sh /usr/local/bin/rocm-llama-entrypoint
RUN chmod +x /usr/local/bin/rocm-llama-entrypoint

EXPOSE 8080

ENTRYPOINT ["rocm-llama-entrypoint"]
