# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    libsndfile1 \
    libvulkan-dev \
    git \
    ninja-build \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Optional ROCm wheel overrides (useful for AMD GPU installs)
ARG ROCM_WHL_URL
ARG ROCM_TORCH_VERSION
ARG ROCM_TORCHAUDIO_VERSION
ARG ROCM_TORCHVISION_VERSION
ARG ROCM_NIGHTLY
ARG LLAMA_BUILD_VULKAN
ARG LLAMA_BUILD_HIPBLAS
ARG LLAMA_CPP_PYTHON_REPO
ARG LLAMA_CPP_PYTHON_REF=v0.3.33
ARG LLAMA_CPP_REPO
ARG LLAMA_CPP_REF=b9888
ARG INSTALL_NEMO

# Copy dependency files and README
COPY pyproject.toml uv.lock README.md ./

# Export dependencies from the committed lock for repeatable builds
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --format requirements.txt -o requirements.txt

# We'll install llama-cpp-python separately to ensure a llama.cpp build that supports Qwen3VL
RUN sed -i '/llama-cpp-python/d' requirements.txt

# Install dependencies with uv (optionally build llama.cpp with Vulkan/HIPBLAS)
RUN set -e; \
    CMAKE_FLAGS=""; \
    if [ "$LLAMA_BUILD_VULKAN" = "1" ]; then \
      CMAKE_FLAGS="$CMAKE_FLAGS -DLLAMA_VULKAN=on -DGGML_VULKAN=on"; \
    fi; \
    if [ "$LLAMA_BUILD_HIPBLAS" = "1" ]; then \
      CMAKE_FLAGS="$CMAKE_FLAGS -DLLAMA_HIPBLAS=on -DGGML_HIPBLAS=on"; \
    fi; \
    if [ -n "$CMAKE_FLAGS" ]; then \
      export CMAKE_ARGS="$CMAKE_FLAGS"; \
      export FORCE_CMAKE=1; \
    fi; \
    uv pip install --system -r requirements.txt

# Install llama-cpp-python with an up-to-date llama.cpp that supports Qwen3VL
RUN set -e; \
    CMAKE_FLAGS=""; \
    if [ "$LLAMA_BUILD_VULKAN" = "1" ]; then \
      CMAKE_FLAGS="$CMAKE_FLAGS -DLLAMA_VULKAN=on -DGGML_VULKAN=on"; \
    fi; \
    if [ "$LLAMA_BUILD_HIPBLAS" = "1" ]; then \
      CMAKE_FLAGS="$CMAKE_FLAGS -DLLAMA_HIPBLAS=on -DGGML_HIPBLAS=on"; \
    fi; \
    if [ -n "$CMAKE_FLAGS" ]; then \
      export CMAKE_ARGS="$CMAKE_FLAGS"; \
      export FORCE_CMAKE=1; \
    fi; \
    repo="${LLAMA_CPP_PYTHON_REPO:-https://github.com/abetlen/llama-cpp-python.git}"; \
    ref="${LLAMA_CPP_PYTHON_REF:-v0.3.33}"; \
    llama_repo="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"; \
    llama_ref="${LLAMA_CPP_REF:-b9888}"; \
    git clone --depth 1 --branch "$ref" "$repo" /tmp/llama-cpp-python; \
    rm -rf /tmp/llama-cpp-python/vendor/llama.cpp; \
    git clone --depth 1 "$llama_repo" /tmp/llama-cpp-python/vendor/llama.cpp; \
    if [ -n "$llama_ref" ] && [ "$llama_ref" != "master" ]; then \
      git -C /tmp/llama-cpp-python/vendor/llama.cpp fetch --depth 1 origin "$llama_ref"; \
      git -C /tmp/llama-cpp-python/vendor/llama.cpp checkout FETCH_HEAD; \
    fi; \
    { \
      echo 'if (NOT DEFINED LLAMA_INSTALL_VERSION OR "${LLAMA_INSTALL_VERSION}" STREQUAL "")'; \
      echo '  set(LLAMA_INSTALL_VERSION 0.0.0)'; \
      echo 'endif()'; \
      cat /tmp/llama-cpp-python/vendor/llama.cpp/tools/mtmd/CMakeLists.txt; \
    } > /tmp/mtmd.CMakeLists.txt && \
    mv /tmp/mtmd.CMakeLists.txt /tmp/llama-cpp-python/vendor/llama.cpp/tools/mtmd/CMakeLists.txt; \
    uv pip install --system /tmp/llama-cpp-python; \
    rm -rf /tmp/llama-cpp-python

RUN python - <<'PY'
from pathlib import Path
import re

path = Path("/usr/local/lib/python3.11/site-packages/llama_cpp/_ctypes_extensions.py")
if path.exists():
    text = path.read_text()
    if "Missing llama.cpp symbol" not in text:
        text, count = re.subn(
            r"(?m)^(?P<indent>[ ]*)func = getattr\(lib, name\)$",
            "\\g<indent>try:\\n"
            "\\g<indent>    func = getattr(lib, name)\\n"
            "\\g<indent>except AttributeError:\\n"
            "\\g<indent>    def missing(*args, **kwargs):\\n"
            "\\g<indent>        raise AttributeError(f\"Missing llama.cpp symbol: {name}\")\\n"
            "\\g<indent>    return missing",
            text,
            count=1,
        )
        if count == 0:
            raise SystemExit(f"Failed to patch ctypes loader at {path}")
        path.write_text(text)
PY

RUN python - <<'PY'
import llama_cpp

print("llama_cpp import ok", getattr(llama_cpp, "__version__", "unknown"))
PY

# Force ROCm PyTorch wheels when configured
RUN if [ -n "$ROCM_WHL_URL" ]; then \
      if [ "$ROCM_NIGHTLY" = "1" ]; then \
        python -m pip install --no-cache-dir --pre --index-url "$ROCM_WHL_URL" \
          torch torchvision torchaudio || \
        python -m pip install --no-cache-dir --force-reinstall \
          --find-links "$ROCM_WHL_URL" \
          "torch==${ROCM_TORCH_VERSION}" \
          "torchaudio==${ROCM_TORCHAUDIO_VERSION}" \
          "torchvision==${ROCM_TORCHVISION_VERSION}"; \
      elif [ -n "$ROCM_TORCH_VERSION" ]; then \
        python -m pip install --no-cache-dir --force-reinstall \
          --find-links "$ROCM_WHL_URL" \
          "torch==${ROCM_TORCH_VERSION}" \
          "torchaudio==${ROCM_TORCHAUDIO_VERSION}" \
          "torchvision==${ROCM_TORCHVISION_VERSION}"; \
      fi; \
    fi

# Optional: install NeMo ASR stack for Canary models
RUN if [ "$INSTALL_NEMO" = "1" ]; then \
      uv pip install --system "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"; \
    fi

# Copy application code (needed for editable install)
COPY app ./app

# Install the local package without re-installing dependencies
RUN uv pip install --system -e . --no-deps

# Create data directories
RUN mkdir -p /data/uploads /data/models /data/chroma

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
