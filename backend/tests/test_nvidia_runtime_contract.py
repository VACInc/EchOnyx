from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_cuda_dockerfile_uses_bundled_llama_vendor_by_default():
    dockerfile_text = (ROOT / "backend" / "Dockerfile.cuda").read_text(encoding="utf-8")

    assert "ARG LLAMA_CPP_USE_BUNDLED_VENDOR=1" in dockerfile_text
    assert 'if [ "$use_bundled_vendor" != "1" ] || [ -n "${LLAMA_CPP_REPO:-}" ] || [ -n "${LLAMA_CPP_REF:-}" ]; then' in dockerfile_text
    assert "git clone --depth 1 --branch \"$ref\" \"$repo\" /tmp/llama-cpp-python" in dockerfile_text
    assert "rm -rf /tmp/llama-cpp-python/vendor/llama.cpp" in dockerfile_text
