from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _service_block(compose_text: str, service: str) -> str:
    start = compose_text.index(f"  {service}:")
    rest = compose_text[start + 2 :]
    end = len(rest)
    for line in rest.splitlines(keepends=True)[1:]:
        if line[:3].strip() and not line.startswith("   "):
            end = rest.index(line)
            break
    return rest[:end]


def test_nvidia_compose_default_vision_is_native_llama_server():
    compose_text = (ROOT / "docker-compose.nvidia.yml").read_text(encoding="utf-8")
    vision = _service_block(compose_text, "vision-server")

    assert "MODEL_RUNTIME=llama_server" in vision
    assert "LLAMA_SERVER_BIN=/opt/cuda-llama/bin/llama-server" in vision
    assert "MODEL_CANDIDATES_JSON" in vision
    assert "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf" in vision
    # Explicit overrides must default empty so candidates stay in control.
    assert "MODEL_PATH=${VISION_MODEL_PATH:-}" in vision
    assert "vllm" not in vision.split("vision-server-vllm")[0].lower()


def test_nvidia_compose_vllm_profile_and_cuda_ordering():
    compose_text = (ROOT / "docker-compose.nvidia.yml").read_text(encoding="utf-8")
    vllm = _service_block(compose_text, "vision-server-vllm")

    assert 'profiles: ["vision-vllm"]' in vllm
    assert "MODEL_RUNTIME=vllm" in vllm
    assert "--served-model-name" in vllm

    assert 'image: ${NVIDIA_VISION_VLLM_IMAGE:-vllm/vllm-openai:v0.17.1}' in compose_text
    assert "exec python3 /opt/echonyx-runtime/managed_openai_runtime.py" in compose_text
    assert "AUTO_NVIDIA_GPU_SELECTION=1" in compose_text
    assert "SERVICE_ROLE=vision" in compose_text
    assert 'NVIDIA_SUMMARIZATION_VISIBLE_DEVICES' in compose_text
    assert 'vllm_cache:/root/.cache/huggingface' in compose_text
    assert "./backend/app/runtime:/opt/echonyx-runtime:ro" in compose_text
    assert "urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" in compose_text
    assert "MODEL_RUNTIME=command" in compose_text
    assert "MODEL_COMMAND=python -m app.runtime.llama_cpp_server" in compose_text
    assert "MODEL_PATH=${SUMMARIZATION_MODEL_PATH:-/models/Qwen3-30B-A3B-Q4_K_M.gguf}" in compose_text
    assert "MODEL_NAME=${SUMMARIZATION_ENDPOINT_MODEL:-Qwen3-30B-A3B-Q4_K_M.gguf}" in compose_text
    assert "SUMMARIZATION_ENDPOINT_MODEL=${SUMMARIZATION_ENDPOINT_MODEL:-Qwen3-30B-A3B-Q4_K_M.gguf}" in compose_text
    assert "model_cache:/models:ro" in compose_text
    assert "qwen3-30b-a3b-q4_k_m.gguf" not in compose_text


def test_cuda_dockerfile_builds_pinned_native_llama_server():
    dockerfile_text = (ROOT / "backend" / "Dockerfile.cuda").read_text(encoding="utf-8")

    assert "LLAMA_SERVER_COMMIT=08f3f4a8a30633491b031bf833441de2a1ab5029" in dockerfile_text
    assert 'test "$(git -C /tmp/llama-server-src rev-parse HEAD)" = "$LLAMA_SERVER_COMMIT"' in dockerfile_text
    assert "/opt/cuda-llama/bin/llama-server" in dockerfile_text


def test_cuda_dockerfile_uses_bundled_llama_vendor_by_default():
    dockerfile_text = (ROOT / "backend" / "Dockerfile.cuda").read_text(encoding="utf-8")

    assert "ARG LLAMA_CPP_USE_BUNDLED_VENDOR=1" in dockerfile_text
    assert 'if [ "$use_bundled_vendor" != "1" ] || [ -n "${LLAMA_CPP_REPO:-}" ] || [ -n "${LLAMA_CPP_REF:-}" ]; then' in dockerfile_text
    assert "git clone --depth 1 --branch \"$ref\" \"$repo\" /tmp/llama-cpp-python" in dockerfile_text
    assert "git -C /tmp/llama-cpp-python submodule update --init --depth 1 vendor/llama.cpp" in dockerfile_text
    assert "rm -rf /tmp/llama-cpp-python/vendor/llama.cpp" in dockerfile_text
