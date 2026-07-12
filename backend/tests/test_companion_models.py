from app.core.model_downloader import MODEL_REGISTRY, companion_model_names


def test_qwen3vl_vision_gguf_declares_its_mmproj():
    assert companion_model_names("Qwen3VL-32B-Instruct-Q4_K_M.gguf") == [
        "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf"
    ]


def test_qwen25vl_vision_gguf_declares_its_mmproj():
    assert companion_model_names("Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf") == [
        "Qwen2.5-VL-3B-Instruct.mmproj-fp16.gguf"
    ]


def test_lookup_is_case_insensitive_and_returns_canonical_names():
    assert companion_model_names("qwen3vl-32b-instruct-q4_k_m.gguf") == [
        "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf"
    ]


def test_models_without_companions_return_empty():
    assert companion_model_names("Qwen3-30B-A3B-Q4_K_M.gguf") == []
    assert companion_model_names("not-in-registry.gguf") == []


def test_declared_companions_resolve_in_registry():
    for name, info in MODEL_REGISTRY.items():
        for companion in info.get("companions", ()):
            assert companion in MODEL_REGISTRY, (
                f"{name} declares companion {companion} that is not registered"
            )
            assert companion != name
