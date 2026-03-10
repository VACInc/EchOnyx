from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_amd_compose_uses_rocm_llama_server_image_contract():
    compose_text = (ROOT / "docker-compose.amd.yml").read_text(encoding="utf-8")

    assert "ghcr.io/ggml-org/llama.cpp:server" not in compose_text
    assert "docker/rocm-openai-runtime/Dockerfile" in compose_text
    assert "repo.radeon.com/rocm/llama.cpp/linux/rocm-rel-7.2/" in compose_text
    assert "VLLM_PIP_SPEC" in compose_text
    assert "--pool=solo" in compose_text
    assert "MODEL_RUNTIME=${ROCM_LLM_RUNTIME:-llama_server}" in compose_text
    assert "IDLE_TIMEOUT_SECONDS=${ROCM_LLM_IDLE_TIMEOUT_S:-120}" in compose_text
    assert "MIOPEN_DEBUG_GCN_ASM_KERNELS=${MIOPEN_DEBUG_GCN_ASM_KERNELS:-0}" in compose_text


def test_rocm_llama_server_files_fail_closed_without_rocm():
    dockerfile_text = (ROOT / "docker/rocm-openai-runtime/Dockerfile").read_text(encoding="utf-8")
    entrypoint_text = (ROOT / "docker/rocm-openai-runtime/entrypoint.sh").read_text(encoding="utf-8")

    assert "rocm/dev-ubuntu-24.04:7.2-complete" in dockerfile_text
    assert "pip install --no-cache-dir --break-system-packages ${VLLM_PIP_SPEC}" in dockerfile_text
    assert "rocminfo" in entrypoint_text
    assert "/dev/kfd" in entrypoint_text
    assert "LD_LIBRARY_PATH" in entrypoint_text
    assert "refusing to start CPU fallback" in entrypoint_text
