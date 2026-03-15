from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_nvidia_compose_uses_vllm_for_vision_and_cuda_ordering():
    compose_text = (ROOT / "docker-compose.nvidia.yml").read_text(encoding="utf-8")

    assert 'image: ${NVIDIA_VISION_VLLM_IMAGE:-vllm/vllm-openai:v0.11.2}' in compose_text
    assert "exec python3 /opt/echonyx-runtime/managed_openai_runtime.py" in compose_text
    assert "MODEL_RUNTIME=vllm" in compose_text
    assert "AUTO_NVIDIA_GPU_SELECTION=1" in compose_text
    assert "SERVICE_ROLE=vision" in compose_text
    assert 'NVIDIA_SUMMARIZATION_VISIBLE_DEVICES' in compose_text
    assert 'vllm_cache:/root/.cache/huggingface' in compose_text
    assert "./backend/app/runtime:/opt/echonyx-runtime:ro" in compose_text
    assert "urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" in compose_text
    assert "MODEL_RUNTIME=command" in compose_text
    assert "MODEL_COMMAND=python -m app.runtime.llama_cpp_server" in compose_text


def test_cuda_dockerfile_uses_bundled_llama_vendor_by_default():
    dockerfile_text = (ROOT / "backend" / "Dockerfile.cuda").read_text(encoding="utf-8")

    assert "ARG LLAMA_CPP_USE_BUNDLED_VENDOR=1" in dockerfile_text
    assert 'if [ "$use_bundled_vendor" != "1" ] || [ -n "${LLAMA_CPP_REPO:-}" ] || [ -n "${LLAMA_CPP_REF:-}" ]; then' in dockerfile_text
    assert "git clone --depth 1 --branch \"$ref\" \"$repo\" /tmp/llama-cpp-python" in dockerfile_text
    assert "git -C /tmp/llama-cpp-python submodule update --init --depth 1 vendor/llama.cpp" in dockerfile_text
    assert "rm -rf /tmp/llama-cpp-python/vendor/llama.cpp" in dockerfile_text
