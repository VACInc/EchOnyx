import subprocess

from app.runtime.managed_openai_runtime import (
    ManagedRuntime,
    RuntimeConfig,
    build_runtime_command,
)


class FakeProcess:
    def __init__(self):
        self.pid = 4321
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


def _runtime_config(**overrides):
    base = dict(
        runtime="llama_server",
        model_command="",
        model_path="/models/model.gguf",
        model_name="model.gguf",
        vllm_model_id="",
        service_role="",
        auto_nvidia_gpu_selection=False,
        model_memory_gb=0.0,
        peer_model_memory_gb=0.0,
        hot_set_memory_gb=0.0,
        shutdown_after_request=False,
        host="0.0.0.0",
        public_port=8080,
        upstream_port=18080,
        context_size=8192,
        gpu_layers=999,
        mmproj_path="",
        idle_timeout_seconds=120,
        startup_timeout_seconds=600,
        proxy_timeout_seconds=600,
        llama_extra_args=[],
        vllm_extra_args=[],
    )
    base.update(overrides)
    return RuntimeConfig(**base)


def test_build_runtime_command_for_llama_server(monkeypatch):
    monkeypatch.setattr(
        "app.runtime.managed_openai_runtime._discover_llama_server_bin",
        lambda: "/opt/amd-llama/bin/llama-server",
    )
    config = _runtime_config(
        mmproj_path="/models/mmproj.gguf",
        llama_extra_args=["--jinja", "--temp", "0.2"],
    )

    command = build_runtime_command(config)

    assert command[:4] == ["/opt/amd-llama/bin/llama-server", "-m", "/models/model.gguf", "-c"]
    assert "--mmproj" in command
    assert "--host" in command
    assert "127.0.0.1" in command
    assert command[-3:] == ["--jinja", "--temp", "0.2"]


def test_build_runtime_command_for_vllm():
    config = _runtime_config(
        runtime="vllm",
        model_path="/models/unused.gguf",
        model_name="summary-model",
        vllm_model_id="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        vllm_extra_args=["--dtype", "auto"],
    )

    command = build_runtime_command(config)

    assert command[:3] == ["vllm", "serve", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"]
    assert "--served-model-name" in command
    assert "summary-model" in command
    assert command[-2:] == ["--dtype", "auto"]


def test_build_runtime_command_for_custom_command():
    config = _runtime_config(
        runtime="command",
        model_command="python -m app.runtime.llama_cpp_server",
        model_path="",
        model_name="summary-model",
    )

    command = build_runtime_command(config)

    assert command == ["python", "-m", "app.runtime.llama_cpp_server"]


def test_build_runtime_command_normalizes_vllm_mm_limits():
    config = _runtime_config(
        runtime="vllm",
        model_path="/models/unused.gguf",
        model_name="vision-model",
        vllm_model_id="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        vllm_extra_args=["--limit-mm-per-prompt", "video=0", "image=4", "--max-num-seqs", "1"],
    )

    command = build_runtime_command(config)

    assert "--limit-mm-per-prompt" in command
    mm_limit_index = command.index("--limit-mm-per-prompt")
    assert command[mm_limit_index + 1] == '{"video":0,"image":4}'
    assert command[-2:] == ["--max-num-seqs", "1"]


def test_runtime_config_accepts_vllm_model_id_without_model_path(monkeypatch):
    monkeypatch.setenv("MODEL_RUNTIME", "vllm")
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("VLLM_MODEL_ID", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")
    monkeypatch.setenv("MODEL_NAME", "vision-endpoint")

    config = RuntimeConfig.from_env()

    assert config.runtime == "vllm"
    assert config.model_path == ""
    assert config.vllm_model_id == "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    assert config.model_name == "vision-endpoint"


def test_managed_runtime_prefers_secondary_gpu_for_summary_when_hot_set_does_not_fit(monkeypatch):
    process = FakeProcess()
    captured = {}

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="5, RTX 6000, 40960\n0, RTX 3090, 24576\n1, RTX 3090, 24576\n",
            stderr="",
        )

    def fake_popen(command, env, start_new_session):
        captured["env"] = env
        return process

    monkeypatch.setattr("app.runtime.managed_openai_runtime.subprocess.run", fake_run)

    runtime = ManagedRuntime(
        _runtime_config(
            runtime="command",
            model_command="python -m app.runtime.llama_cpp_server",
            model_path="",
            service_role="summarization",
            auto_nvidia_gpu_selection=True,
            model_memory_gb=24.0,
            peer_model_memory_gb=62.0,
            hot_set_memory_gb=86.0,
        ),
        popen_factory=fake_popen,
        health_check=lambda _config: False,
    )

    runtime.ensure_started()

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_managed_runtime_uses_role_specific_explicit_gpu_pin(monkeypatch):
    process = FakeProcess()
    captured = {}

    def fake_popen(command, env, start_new_session):
        captured["env"] = env
        return process

    monkeypatch.setenv("NVIDIA_VISION_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MODEL_VISIBLE_DEVICES", raising=False)

    runtime = ManagedRuntime(
        _runtime_config(
            runtime="command",
            model_command="python -m app.runtime.llama_cpp_server",
            model_path="",
            service_role="summarization",
            auto_nvidia_gpu_selection=True,
        ),
        popen_factory=fake_popen,
        health_check=lambda _config: False,
    )

    runtime.ensure_started()

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_managed_runtime_command_child_uses_upstream_port(monkeypatch):
    process = FakeProcess()
    captured = {}

    def fake_popen(command, env, start_new_session):
        captured["env"] = env
        return process

    runtime = ManagedRuntime(
        _runtime_config(
            runtime="command",
            model_command="python -m app.runtime.llama_cpp_server",
            model_path="",
            service_role="summarization",
            public_port=8000,
            upstream_port=18080,
        ),
        popen_factory=fake_popen,
        health_check=lambda _config: False,
    )

    runtime.ensure_started()

    assert captured["env"]["PORT"] == "18080"
    assert captured["env"]["LISTEN_HOST"] == "127.0.0.1"


def test_managed_runtime_enables_shutdown_after_request_on_single_small_gpu(monkeypatch):
    process = FakeProcess()
    captured = {}

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0, RTX 4090, 24576\n",
            stderr="",
        )

    def fake_popen(command, env, start_new_session):
        captured["env"] = env
        return process

    monkeypatch.setattr("app.runtime.managed_openai_runtime.subprocess.run", fake_run)

    runtime = ManagedRuntime(
        _runtime_config(
            runtime="command",
            model_command="python -m app.runtime.llama_cpp_server",
            model_path="",
            service_role="summarization",
            auto_nvidia_gpu_selection=True,
            model_memory_gb=24.0,
            peer_model_memory_gb=62.0,
            hot_set_memory_gb=86.0,
        ),
        popen_factory=fake_popen,
        health_check=lambda _config: True,
    )

    runtime.ensure_started()
    runtime._process = process
    runtime._process_group_id = process.pid
    runtime.request_started()

    monkeypatch.setattr("app.runtime.managed_openai_runtime.os.killpg", lambda *_args: setattr(process, "returncode", 0))
    runtime.request_finished()

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert runtime.child_running() is False


def test_managed_runtime_starts_once_until_child_is_ready(monkeypatch):
    process = FakeProcess()
    popen_calls = []
    health_checks = iter([False, True])
    current_time = {"value": 100.0}

    def fake_clock():
        return current_time["value"]

    def fake_popen(command, env, start_new_session):
        popen_calls.append((command, env, start_new_session))
        return process

    monkeypatch.setattr(
        "app.runtime.managed_openai_runtime._discover_llama_server_bin",
        lambda: "/opt/amd-llama/bin/llama-server",
    )

    runtime = ManagedRuntime(
        _runtime_config(),
        popen_factory=fake_popen,
        clock=fake_clock,
        health_check=lambda _config: next(health_checks),
    )

    assert runtime.ensure_started() is False
    assert runtime.ensure_started() is False
    assert runtime.ensure_started() is True
    assert len(popen_calls) == 1
    assert popen_calls[0][2] is True


def test_managed_runtime_stops_idle_process(monkeypatch):
    process = FakeProcess()
    current_time = {"value": 100.0}
    killpg_calls = []

    def fake_killpg(pid, sig):
        killpg_calls.append((pid, sig))
        process.returncode = 0

    runtime = ManagedRuntime(
        _runtime_config(idle_timeout_seconds=30),
        popen_factory=lambda *_args, **_kwargs: process,
        clock=lambda: current_time["value"],
        health_check=lambda _config: True,
    )

    monkeypatch.setattr("app.runtime.managed_openai_runtime.os.killpg", fake_killpg)
    runtime._process = process
    runtime._process_group_id = process.pid
    runtime._last_request_ts = 100.0
    current_time["value"] = 131.0

    assert runtime.maybe_stop_idle_process() is True
    assert killpg_calls
    assert runtime.child_running() is False


def test_managed_runtime_does_not_stop_when_idle_timeout_disabled():
    process = FakeProcess()

    runtime = ManagedRuntime(
        _runtime_config(idle_timeout_seconds=0),
        popen_factory=lambda *_args, **_kwargs: process,
        clock=lambda: 500.0,
        health_check=lambda _config: True,
    )

    runtime._process = process
    runtime._last_request_ts = 100.0

    assert runtime.maybe_stop_idle_process() is False
    assert process.terminated is False


def test_managed_runtime_does_not_stop_while_request_is_in_flight():
    process = FakeProcess()

    runtime = ManagedRuntime(
        _runtime_config(idle_timeout_seconds=30),
        popen_factory=lambda *_args, **_kwargs: process,
        clock=lambda: 500.0,
        health_check=lambda _config: True,
    )

    runtime._process = process
    runtime._last_request_ts = 100.0
    runtime._active_requests = 1

    assert runtime.maybe_stop_idle_process() is False
    assert process.terminated is False
