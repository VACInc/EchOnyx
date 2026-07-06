import os
from pathlib import Path

import pytest

from app.runtime import llama_cpp_server


def test_build_server_env_downloads_missing_files(monkeypatch, tmp_path):
    model_path = tmp_path / "Qwen3VL-32B-Instruct-Q4_K_M.gguf"
    mmproj_path = tmp_path / "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf"
    downloaded = []

    def fake_download(name: str, cache_dir: Path, **_kwargs) -> Path:
        path = cache_dir / name
        path.write_text("stub")
        downloaded.append(path.name)
        return path

    monkeypatch.setattr(llama_cpp_server, "download_model", fake_download)

    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=model_path,
        model_alias="vision-model",
        host="0.0.0.0",
        port=8000,
        context_size=8192,
        gpu_layers=999,
        main_gpu=None,
        split_mode=None,
        chat_format="qwen3-vl",
        clip_model_path=mmproj_path,
        extra_args=("--verbose",),
    )

    env = llama_cpp_server.build_server_env(config)

    assert downloaded == [model_path.name, mmproj_path.name]
    assert env["MODEL"] == str(model_path)
    assert env["CLIP_MODEL_PATH"] == str(mmproj_path)
    assert env["MODEL_ALIAS"] == "vision-model"
    assert env["CHAT_FORMAT"] == "qwen3-vl"
    assert env["N_GPU_LAYERS"] == "999"


def test_build_server_env_resolves_bare_model_name_under_model_cache_dir(monkeypatch, tmp_path):
    model_path = tmp_path / "Qwen3-30B-A3B-Q4_K_M.gguf"
    model_path.write_text("stub")
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path))

    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=Path(model_path.name),
        model_alias="summary-model",
        host="0.0.0.0",
        port=8000,
        context_size=32768,
        gpu_layers=999,
        main_gpu=None,
        split_mode=None,
        chat_format="",
        clip_model_path=None,
        extra_args=(),
    )

    env = llama_cpp_server.build_server_env(config)

    assert env["MODEL"] == str(model_path)


def test_build_server_env_uses_case_insensitive_existing_file(monkeypatch, tmp_path):
    configured_path = tmp_path / "Qwen3-30B-A3B-Q4_K_M.gguf"
    existing_path = tmp_path / "qwen3-30b-a3b-q4_k_m.gguf"
    existing_path.write_text("stub")
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path))

    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=configured_path,
        model_alias="summary-model",
        host="0.0.0.0",
        port=8000,
        context_size=32768,
        gpu_layers=999,
        main_gpu=None,
        split_mode=None,
        chat_format="",
        clip_model_path=None,
        extra_args=(),
    )

    env = llama_cpp_server.build_server_env(config)

    assert env["MODEL"] == str(existing_path)


def test_build_server_env_fails_missing_file_when_auto_download_disabled(monkeypatch, tmp_path):
    model_path = tmp_path / "Qwen3VL-32B-Instruct-Q4_K_M.gguf"
    monkeypatch.setenv("MODEL_AUTO_DOWNLOAD", "false")

    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=model_path,
        model_alias="vision-model",
        host="0.0.0.0",
        port=8000,
        context_size=8192,
        gpu_layers=999,
        main_gpu=None,
        split_mode=None,
        chat_format="qwen3-vl",
        clip_model_path=None,
        extra_args=(),
    )

    with pytest.raises(RuntimeError, match="Download it in Settings or set MODEL_AUTO_DOWNLOAD=true"):
        llama_cpp_server.build_server_env(config)


def test_build_server_env_infers_single_cuda_visible_device(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_text("stub")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.delenv("NVIDIA_VISION_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", raising=False)

    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=model_path,
        model_alias="summary-model",
        host="0.0.0.0",
        port=8000,
        context_size=8192,
        gpu_layers=999,
        main_gpu=None,
        split_mode=None,
        chat_format="",
        clip_model_path=None,
        extra_args=(),
    )

    env = llama_cpp_server.build_server_env(config)

    assert env["MAIN_GPU"] == "0"
    assert env["SPLIT_MODE"] == "0"


def test_build_server_env_infers_single_cuda_visible_uuid(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_text("stub")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-8003976c-76b3-45e1-69aa-181bf5e07b05")
    monkeypatch.delenv("NVIDIA_VISION_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", raising=False)

    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=model_path,
        model_alias="summary-model",
        host="0.0.0.0",
        port=8000,
        context_size=8192,
        gpu_layers=999,
        main_gpu=None,
        split_mode=None,
        chat_format="",
        clip_model_path=None,
        extra_args=(),
    )

    env = llama_cpp_server.build_server_env(config)

    assert env["MAIN_GPU"] == "0"
    assert env["SPLIT_MODE"] == "0"


def test_build_server_env_infers_single_nvidia_pin_when_all_gpus_visible(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_text("stub")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("NVIDIA_VISION_VISIBLE_DEVICES", "5")

    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=model_path,
        model_alias="vision-model",
        host="0.0.0.0",
        port=8000,
        context_size=8192,
        gpu_layers=999,
        main_gpu=None,
        split_mode=None,
        chat_format="qwen3-vl",
        clip_model_path=None,
        extra_args=(),
    )

    env = llama_cpp_server.build_server_env(config)

    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert env["MAIN_GPU"] == "0"
    assert env["SPLIT_MODE"] == "0"


def test_build_server_env_translates_single_host_pin_to_local_main_gpu(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_text("stub")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", "4")

    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=model_path,
        model_alias="summary-model",
        host="0.0.0.0",
        port=8000,
        context_size=8192,
        gpu_layers=999,
        main_gpu=None,
        split_mode=None,
        chat_format="",
        clip_model_path=None,
        extra_args=(),
    )

    env = llama_cpp_server.build_server_env(config)

    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["MAIN_GPU"] == "0"
    assert env["SPLIT_MODE"] == "0"


def test_build_server_env_translates_selected_host_gpu_to_visible_local_ordinal(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_text("stub")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,4")

    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=model_path,
        model_alias="summary-model",
        host="0.0.0.0",
        port=8000,
        context_size=8192,
        gpu_layers=999,
        main_gpu=4,
        split_mode=None,
        chat_format="",
        clip_model_path=None,
        extra_args=(),
    )

    env = llama_cpp_server.build_server_env(config)

    assert env["CUDA_VISIBLE_DEVICES"] == "1,4"
    assert env["MAIN_GPU"] == "1"
    assert env["SPLIT_MODE"] == "0"


def test_build_server_env_prefers_explicit_model_main_gpu(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_text("stub")
    monkeypatch.setenv("NVIDIA_VISION_VISIBLE_DEVICES", "5")

    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=model_path,
        model_alias="vision-model",
        host="0.0.0.0",
        port=8000,
        context_size=8192,
        gpu_layers=999,
        main_gpu=2,
        split_mode=None,
        chat_format="qwen3-vl",
        clip_model_path=None,
        extra_args=(),
    )

    env = llama_cpp_server.build_server_env(config)

    assert env["MAIN_GPU"] == "2"
    assert env["SPLIT_MODE"] == "0"


def test_build_server_env_narrows_pinned_host_index_under_parent_visibility(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_text("stub")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,4,6")
    monkeypatch.setenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", "4")
    monkeypatch.delenv("NVIDIA_VISION_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MODEL_VISIBLE_DEVICES", raising=False)

    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=model_path,
        model_alias="summary-model",
        host="0.0.0.0",
        port=8000,
        context_size=8192,
        gpu_layers=999,
        main_gpu=None,
        split_mode=None,
        chat_format="",
        clip_model_path=None,
        extra_args=(),
    )

    env = llama_cpp_server.build_server_env(config)

    # CUDA_VISIBLE_DEVICES does not compose: exporting a new value re-enumerates
    # physical devices, so the pin is kept verbatim as the physical id.
    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["MAIN_GPU"] == "0"
    assert env["SPLIT_MODE"] == "0"


def test_build_server_env_exports_host_pin_verbatim_when_parent_unset(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_text("stub")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", "4")
    monkeypatch.delenv("NVIDIA_VISION_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MODEL_VISIBLE_DEVICES", raising=False)

    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=model_path,
        model_alias="summary-model",
        host="0.0.0.0",
        port=8000,
        context_size=8192,
        gpu_layers=999,
        main_gpu=None,
        split_mode=None,
        chat_format="",
        clip_model_path=None,
        extra_args=(),
    )

    env = llama_cpp_server.build_server_env(config)

    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["MAIN_GPU"] == "0"
    assert env["SPLIT_MODE"] == "0"


def test_build_server_env_fails_fast_when_pin_not_resolvable(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_text("stub")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,4,6")
    monkeypatch.setenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", "7")
    monkeypatch.delenv("NVIDIA_VISION_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MODEL_VISIBLE_DEVICES", raising=False)

    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=model_path,
        model_alias="summary-model",
        host="0.0.0.0",
        port=8000,
        context_size=8192,
        gpu_layers=999,
        main_gpu=None,
        split_mode=None,
        chat_format="",
        clip_model_path=None,
        extra_args=(),
    )

    with pytest.raises(RuntimeError, match="visible CUDA devices"):
        llama_cpp_server.build_server_env(config)


def test_build_server_command_uses_llama_cpp_server_module():
    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=Path("/data/models/model.gguf"),
        model_alias="summary-model",
        host="0.0.0.0",
        port=8000,
        context_size=32768,
        gpu_layers=999,
        main_gpu=None,
        split_mode=None,
        chat_format="",
        clip_model_path=None,
        extra_args=("--slots",),
    )

    command = llama_cpp_server.build_server_command(config)

    assert command[:3] == [os.sys.executable, "-m", "llama_cpp.server"]
    assert command[-1] == "--slots"


def test_ensure_server_dependencies_installs_missing_module(monkeypatch):
    monkeypatch.setattr(
        llama_cpp_server.importlib.util,
        "find_spec",
        lambda name: None if name in {"sse_starlette", "starlette_context"} else object(),
    )
    seen = {}

    def fake_check_call(cmd):
        seen["cmd"] = cmd

    monkeypatch.setattr(llama_cpp_server.subprocess, "check_call", fake_check_call)

    llama_cpp_server.ensure_server_dependencies()

    assert seen["cmd"] == [
        os.sys.executable,
        "-m",
        "pip",
        "install",
        "sse-starlette>=2.1.3",
        "starlette-context>=0.3.6",
    ]
