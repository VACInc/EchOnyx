import pytest

from app.core.model_downloader import download_model


def test_download_model_refuses_read_only_target_dir_with_env_guidance(tmp_path):
    read_only_dir = tmp_path / "models"
    read_only_dir.mkdir()
    read_only_dir.chmod(0o555)

    try:
        with pytest.raises(RuntimeError) as exc:
            download_model(
                "Qwen3-30B-A3B-Q4_K_M.gguf",
                read_only_dir,
                env_var="MODEL_PATH",
            )
    finally:
        read_only_dir.chmod(0o755)

    message = str(exc.value)
    assert "Qwen3-30B-A3B-Q4_K_M.gguf" in message
    assert str(read_only_dir / "Qwen3-30B-A3B-Q4_K_M.gguf") in message
    assert "MODEL_PATH" in message
