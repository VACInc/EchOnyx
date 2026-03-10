import subprocess

from app.runtime.managed_openai_runtime import (
    ManagedRuntime,
    RuntimeConfig,
    build_runtime_command,
)


class FakeProcess:
    def __init__(self):
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
        model_path="/models/model.gguf",
        model_name="model.gguf",
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
        model_path="Qwen/Qwen3-30B-A3B-Instruct-2507",
        model_name="summary-model",
        vllm_extra_args=["--dtype", "auto"],
    )

    command = build_runtime_command(config)

    assert command[:3] == ["vllm", "serve", "Qwen/Qwen3-30B-A3B-Instruct-2507"]
    assert "--served-model-name" in command
    assert "summary-model" in command
    assert command[-2:] == ["--dtype", "auto"]


def test_managed_runtime_starts_once_until_child_is_ready(monkeypatch):
    process = FakeProcess()
    popen_calls = []
    health_checks = iter([False, True])
    current_time = {"value": 100.0}

    def fake_clock():
        return current_time["value"]

    def fake_popen(command, env):
        popen_calls.append((command, env))
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


def test_managed_runtime_stops_idle_process():
    process = FakeProcess()
    current_time = {"value": 100.0}

    runtime = ManagedRuntime(
        _runtime_config(idle_timeout_seconds=30),
        popen_factory=lambda *_args, **_kwargs: process,
        clock=lambda: current_time["value"],
        health_check=lambda _config: True,
    )

    runtime._process = process
    runtime._last_request_ts = 100.0
    current_time["value"] = 131.0

    assert runtime.maybe_stop_idle_process() is True
    assert process.terminated is True
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
