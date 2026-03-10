#!/usr/bin/env bash
set -euo pipefail

if [[ ! -e /dev/kfd ]]; then
  echo "ROCm device /dev/kfd is missing; refusing to start CPU fallback." >&2
  exit 1
fi

if [[ ! -e /dev/dri ]]; then
  echo "DRM device /dev/dri is missing; refusing to start CPU fallback." >&2
  exit 1
fi

if ! command -v rocminfo >/dev/null 2>&1; then
  echo "rocminfo is unavailable inside the ROCm runtime image." >&2
  exit 1
fi

rocminfo_output="$(rocminfo 2>/dev/null || true)"
if [[ -z "${rocminfo_output}" ]]; then
  echo "rocminfo could not enumerate a ROCm GPU; refusing to start CPU fallback." >&2
  exit 1
fi

if ! grep -Eq 'gfx(110|115|120|11|12)' <<<"${rocminfo_output}"; then
  echo "No supported gfx11/gfx12 ROCm device was detected." >&2
  exit 1
fi

llama_server_bin="${LLAMA_SERVER_BIN:-$(find /opt/amd-llama -type f -name llama-server -print -quit)}"
if [[ -z "${llama_server_bin}" || ! -x "${llama_server_bin}" ]]; then
  echo "Unable to locate an executable llama-server binary in /opt/amd-llama." >&2
  exit 1
fi

llama_server_dir="$(dirname "${llama_server_bin}")"
export LD_LIBRARY_PATH="${llama_server_dir}:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/bin:${PATH}"

if [[ "${MODEL_RUNTIME:-llama_server}" == "vllm" ]] && ! command -v vllm >/dev/null 2>&1; then
  echo "MODEL_RUNTIME=vllm requires a vLLM-enabled image build (set INSTALL_VLLM=1)." >&2
  exit 1
fi

echo "Starting managed ROCm OpenAI runtime gateway (${MODEL_RUNTIME:-llama_server})" >&2
exec python3 /opt/rocm-openai-runtime/managed_openai_runtime.py
