from app.api.routes.settings import (
    ModelDownloadResponse,
    _companion_gap_entry,
    _missing_companion_names,
)
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


def test_missing_companions_uses_cache_dir(tmp_path):
    vision = "Qwen3VL-32B-Instruct-Q4_K_M.gguf"
    mmproj = "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf"
    assert _missing_companion_names(vision, tmp_path) == [mmproj]

    (tmp_path / mmproj).write_bytes(b"gguf")
    assert _missing_companion_names(vision, tmp_path) == []


def test_companion_gap_reports_cached_primary_as_uncached():
    vision = "Qwen3VL-32B-Instruct-Q4_K_M.gguf"
    mmproj = "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf"
    entry = _companion_gap_entry(vision, "vision", [mmproj], {})

    assert entry == {
        "model_name": vision,
        "status": "uncached",
        "expected_size_gb": 1.0,
        "missing_companions": [mmproj],
    }


def test_companion_gap_prefers_live_download_progress():
    vision = "Qwen3VL-32B-Instruct-Q4_K_M.gguf"
    mmproj = "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf"
    progress = {mmproj: {"model_name": mmproj, "status": "downloading", "progress_percent": 40}}

    entry = _companion_gap_entry(vision, "vision", [mmproj], progress)

    assert entry is not None
    assert entry["status"] == "downloading"


def test_companion_gap_absent_when_companions_present():
    assert _companion_gap_entry("Qwen3VL-32B-Instruct-Q4_K_M.gguf", "vision", [], {}) is None


def test_download_response_schema_declares_companions():
    response = ModelDownloadResponse(
        model_name="Qwen3VL-32B-Instruct-Q4_K_M.gguf",
        status="cached",
        companions=[
            {"model_name": "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf", "status": "error", "detail": "disk full"}
        ],
    )
    dumped = response.model_dump()
    assert dumped["companions"][0]["status"] == "error"
    assert "companions" in ModelDownloadResponse.model_fields
