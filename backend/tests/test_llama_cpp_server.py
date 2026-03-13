import os
from pathlib import Path

from app.runtime import llama_cpp_server


def test_build_server_env_downloads_missing_files(monkeypatch, tmp_path):
    model_path = tmp_path / "Qwen3VL-32B-Instruct-Q4_K_M.gguf"
    mmproj_path = tmp_path / "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf"
    downloaded = []

    def fake_download(name: str, cache_dir: Path) -> Path:
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


def test_build_server_command_uses_llama_cpp_server_module():
    config = llama_cpp_server.LlamaCppServerConfig(
        model_path=Path("/data/models/model.gguf"),
        model_alias="summary-model",
        host="0.0.0.0",
        port=8000,
        context_size=32768,
        gpu_layers=999,
        chat_format="",
        clip_model_path=None,
        extra_args=("--slots",),
    )

    command = llama_cpp_server.build_server_command(config)

    assert command[:3] == [os.sys.executable, "-m", "llama_cpp.server"]
    assert command[-1] == "--slots"
