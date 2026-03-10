from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_amd_compose_uses_rocm_llama_server_image_contract():
    compose_text = (ROOT / "docker-compose.amd.yml").read_text(encoding="utf-8")

    assert "ghcr.io/ggml-org/llama.cpp:server" not in compose_text
    assert "docker/rocm-openai-runtime/Dockerfile" in compose_text
    assert "repo.radeon.com/rocm/llama.cpp/linux/rocm-rel-7.2/" in compose_text
    assert "INSTALL_VLLM" in compose_text
    assert "VLLM_INSTALL_METHOD" in compose_text
    assert "VLLM_MODEL_ID=${VISION_VLLM_MODEL_ID:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}" in compose_text
    assert (
        "VLLM_MODEL_ID=${SUMMARIZATION_VLLM_MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
        in compose_text
    )
    assert "vllm_cache:/cache" in compose_text
    assert "--pool=solo" in compose_text
    assert "MODEL_RUNTIME=${ROCM_LLM_RUNTIME:-llama_server}" in compose_text
    assert "IDLE_TIMEOUT_SECONDS=${ROCM_LLM_IDLE_TIMEOUT_S:-120}" in compose_text
    assert "MIOPEN_DEBUG_GCN_ASM_KERNELS=${MIOPEN_DEBUG_GCN_ASM_KERNELS:-0}" in compose_text


def test_rocm_llama_server_files_fail_closed_without_rocm():
    dockerfile_text = (ROOT / "docker/rocm-openai-runtime/Dockerfile").read_text(encoding="utf-8")
    entrypoint_text = (ROOT / "docker/rocm-openai-runtime/entrypoint.sh").read_text(encoding="utf-8")

    assert "rocm/dev-ubuntu-24.04:7.2-complete" in dockerfile_text
    assert 'if [ "${INSTALL_VLLM}" = "1" ]' in dockerfile_text
    assert 'if [ "${VLLM_INSTALL_METHOD}" = "wheel" ]' in dockerfile_text
    assert "https://wheels.vllm.ai/rocm/" in dockerfile_text
    assert "libopenmpi3t64" in dockerfile_text
    assert "openmpi-bin" in dockerfile_text
    assert "find /opt/amd-llama -type f -name 'llama-*' -exec chmod +x {} +" in dockerfile_text
    assert "python3 use_existing_torch.py" in dockerfile_text
    assert "requirements/rocm.txt" in dockerfile_text
    assert "rocminfo" in entrypoint_text
    assert "/dev/kfd" in entrypoint_text
    assert "LD_LIBRARY_PATH" in entrypoint_text
    assert "MODEL_RUNTIME=vllm requires a vLLM-enabled image build" in entrypoint_text
    assert "refusing to start CPU fallback" in entrypoint_text
