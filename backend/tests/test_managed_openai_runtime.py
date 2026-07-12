import subprocess
import urllib.error

import pytest

from app.runtime.managed_openai_runtime import (
    ManagedRuntime,
    RuntimeConfig,
    TransientStartError,
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


def test_build_runtime_command_translates_llama_tensor_split_from_host_slots(monkeypatch):
    monkeypatch.setattr(
        "app.runtime.managed_openai_runtime._discover_llama_server_bin",
        lambda: "/opt/amd-llama/bin/llama-server",
    )
    config = _runtime_config(
        llama_extra_args=["--tensor-split", "0,0.25,0,0,0.75"],
    )

    command = build_runtime_command(config, cuda_visible_devices="1,4")

    tensor_split_index = command.index("--tensor-split")
    assert command[tensor_split_index + 1] == "0.25,0.75"


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
            stdout="5, RTX 6000, 49152, 8192, 40960, 0\n0, RTX 3090, 24576, 0, 24576, 0\n1, RTX 3090, 24576, 0, 24576, 0\n",
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

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"


def test_managed_runtime_prefers_idle_gpu_that_fits_over_busier_larger_gpu(monkeypatch):
    process = FakeProcess()
    captured = {}

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="5, RTX PRO 6000, 97887, 70142, 27109, 0\n0, RTX 3090, 24576, 1, 24126, 0\n1, RTX 3090, 24576, 1, 24126, 0\n",
            stderr="",
        )

    def fake_popen(command, env, start_new_session):
        captured["env"] = env
        return process

    monkeypatch.setattr("app.runtime.managed_openai_runtime.subprocess.run", fake_run)

    runtime = ManagedRuntime(
        _runtime_config(
            runtime="vllm",
            service_role="vision",
            auto_nvidia_gpu_selection=True,
            model_memory_gb=20.0,
            peer_model_memory_gb=0.0,
            hot_set_memory_gb=20.0,
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


def test_managed_runtime_translates_llama_main_gpu_for_single_host_pin(monkeypatch):
    process = FakeProcess()
    captured = {}

    def fake_popen(command, env, start_new_session):
        captured["command"] = command
        captured["env"] = env
        return process

    monkeypatch.setattr(
        "app.runtime.managed_openai_runtime._discover_llama_server_bin",
        lambda: "/opt/amd-llama/bin/llama-server",
    )
    monkeypatch.setenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", "4")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MODEL_VISIBLE_DEVICES", raising=False)

    runtime = ManagedRuntime(
        _runtime_config(
            runtime="llama_server",
            service_role="summarization",
            llama_extra_args=["--main-gpu", "4"],
        ),
        popen_factory=fake_popen,
        health_check=lambda _config: False,
    )

    runtime.ensure_started()

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "4"
    main_gpu_index = captured["command"].index("--main-gpu")
    assert captured["command"][main_gpu_index + 1] == "0"


def test_managed_runtime_translates_vllm_device_for_second_visible_gpu(monkeypatch):
    process = FakeProcess()
    captured = {}

    def fake_popen(command, env, start_new_session):
        captured["command"] = command
        captured["env"] = env
        return process

    monkeypatch.setenv("NVIDIA_VISION_VISIBLE_DEVICES", "1,4")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MODEL_VISIBLE_DEVICES", raising=False)

    runtime = ManagedRuntime(
        _runtime_config(
            runtime="vllm",
            service_role="vision",
            model_path="/models/unused.gguf",
            vllm_model_id="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
            vllm_extra_args=["--device", "cuda:4"],
        ),
        popen_factory=fake_popen,
        health_check=lambda _config: False,
    )

    runtime.ensure_started()

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1,4"
    device_index = captured["command"].index("--device")
    assert captured["command"][device_index + 1] == "cuda:1"


def test_managed_runtime_narrows_pinned_host_index_under_parent_visibility(monkeypatch):
    process = FakeProcess()
    captured = {}

    def fake_popen(command, env, start_new_session):
        captured["command"] = command
        captured["env"] = env
        return process

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,4,6")
    monkeypatch.setenv("NVIDIA_VISION_VISIBLE_DEVICES", "4")
    monkeypatch.delenv("MODEL_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", raising=False)

    runtime = ManagedRuntime(
        _runtime_config(
            runtime="vllm",
            service_role="vision",
            model_path="/models/unused.gguf",
            vllm_model_id="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
            vllm_extra_args=["--device", "cuda:4"],
        ),
        popen_factory=fake_popen,
        health_check=lambda _config: False,
    )

    runtime.ensure_started()

    # CUDA_VISIBLE_DEVICES does not compose: the child re-enumerates physical
    # devices, so the pin is exported verbatim as the physical id.
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "4"
    # The child only sees host GPU 4, so engine device args map to ordinal 0.
    device_index = captured["command"].index("--device")
    assert captured["command"][device_index + 1] == "cuda:0"


def test_managed_runtime_exports_host_pin_verbatim_when_parent_unset(monkeypatch):
    process = FakeProcess()
    captured = {}

    def fake_popen(command, env, start_new_session):
        captured["command"] = command
        captured["env"] = env
        return process

    monkeypatch.setattr(
        "app.runtime.managed_openai_runtime._discover_llama_server_bin",
        lambda: "/opt/amd-llama/bin/llama-server",
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", "4")
    monkeypatch.delenv("MODEL_VISIBLE_DEVICES", raising=False)

    runtime = ManagedRuntime(
        _runtime_config(
            runtime="llama_server",
            service_role="summarization",
            llama_extra_args=["--main-gpu", "4"],
        ),
        popen_factory=fake_popen,
        health_check=lambda _config: False,
    )

    runtime.ensure_started()

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "4"
    main_gpu_index = captured["command"].index("--main-gpu")
    assert captured["command"][main_gpu_index + 1] == "0"


def test_managed_runtime_fails_fast_when_pin_not_resolvable(monkeypatch):
    process = FakeProcess()

    def fake_popen(command, env, start_new_session):  # pragma: no cover - must not run
        return process

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,4,6")
    monkeypatch.setenv("NVIDIA_VISION_VISIBLE_DEVICES", "7")
    monkeypatch.delenv("MODEL_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", raising=False)

    runtime = ManagedRuntime(
        _runtime_config(
            runtime="vllm",
            service_role="vision",
            model_path="/models/unused.gguf",
            vllm_model_id="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        ),
        popen_factory=fake_popen,
        health_check=lambda _config: False,
    )

    with pytest.raises(RuntimeError, match="visible CUDA devices"):
        runtime.ensure_started()


def test_managed_runtime_auto_pick_translates_host_index_under_parent_visibility(monkeypatch):
    process = FakeProcess()
    captured = {}

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "0, RTX 3090, 24576, 0, 24576, 0\n"
                "1, RTX 3090, 24576, 20000, 4576, 90\n"
                "4, RTX 3090, 24576, 100, 24476, 0\n"
                "6, RTX 3090, 24576, 18000, 6576, 80\n"
            ),
            stderr="",
        )

    def fake_popen(command, env, start_new_session):
        captured["env"] = env
        return process

    monkeypatch.setattr("app.runtime.managed_openai_runtime.subprocess.run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,4,6")
    monkeypatch.delenv("NVIDIA_VISION_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MODEL_VISIBLE_DEVICES", raising=False)

    runtime = ManagedRuntime(
        _runtime_config(
            runtime="vllm",
            service_role="vision",
            model_path="/models/unused.gguf",
            vllm_model_id="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
            auto_nvidia_gpu_selection=True,
            model_memory_gb=20.0,
            hot_set_memory_gb=20.0,
        ),
        popen_factory=fake_popen,
        health_check=lambda _config: False,
    )

    runtime.ensure_started()

    # Idle host GPU 4 (outside-allowlist GPU 0 is filtered out) is exported
    # verbatim as its physical id.
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "4"


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


def test_managed_runtime_health_check_falls_back_to_v1_models_for_command(monkeypatch):
    runtime = ManagedRuntime(_runtime_config(runtime="command", model_command="python -m app.runtime.llama_cpp_server", model_path=""))

    class FakeResponse:
        def __init__(self, status):
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("/health"):
            raise urllib.error.HTTPError(request.full_url, 404, "not found", hdrs=None, fp=None)
        return FakeResponse(200)

    monkeypatch.setattr("app.runtime.managed_openai_runtime.urllib.request.urlopen", fake_urlopen)

    assert runtime._default_health_check(runtime.config) is True
    assert calls == [
        "http://127.0.0.1:18080/health",
        "http://127.0.0.1:18080/v1/models",
    ]


def test_managed_runtime_enables_shutdown_after_request_on_single_small_gpu(monkeypatch):
    process = FakeProcess()
    captured = {}

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0, RTX 4090, 24576, 0, 24576, 0\n",
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
    runtime.note_served()

    monkeypatch.setattr("app.runtime.managed_openai_runtime.os.killpg", lambda *_args: setattr(process, "returncode", 0))
    runtime.request_finished()

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    # Hot-set pressure now swaps via a short idle linger, not per-request
    # teardown: the child survives the request and is reaped once idle.
    assert runtime.child_running() is True
    assert runtime._swap_idle_override_s == 30.0
    runtime._last_request_ts = runtime._clock() - 31.0
    assert runtime.maybe_stop_idle_process() is True
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


def test_unresolvable_pin_is_memoized_as_fatal_and_popen_never_runs(monkeypatch):
    popen_calls = []

    def fake_popen(command, env, start_new_session):
        popen_calls.append(command)
        return FakeProcess()

    monkeypatch.setenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    runtime = ManagedRuntime(
        _runtime_config(service_role="summarization"),
        popen_factory=fake_popen,
        health_check=lambda _config: True,
    )

    assert runtime.fatal_config_error() is None
    with pytest.raises(RuntimeError, match="pinned GPUs"):
        runtime.ensure_started()

    fatal = runtime.fatal_config_error()
    assert fatal is not None and "pinned GPUs" in fatal
    assert popen_calls == []

    # Second attempt short-circuits on the memoized fatal error.
    with pytest.raises(RuntimeError, match="pinned GPUs"):
        runtime.ensure_started()
    assert popen_calls == []


def test_validate_startup_config_flips_health_before_any_request(monkeypatch):
    monkeypatch.setenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    runtime = ManagedRuntime(
        _runtime_config(service_role="summarization"),
        popen_factory=lambda *a, **k: FakeProcess(),
        health_check=lambda _config: True,
    )
    runtime.validate_startup_config()

    snapshot = runtime.health_snapshot()
    assert snapshot["fatal"] is True
    assert snapshot["child_running"] is False


def test_validate_startup_config_ignores_transient_auto_selection(monkeypatch):
    monkeypatch.delenv("NVIDIA_VISION_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    runtime = ManagedRuntime(
        _runtime_config(
            service_role="vision",
            auto_nvidia_gpu_selection=True,
            runtime="command",
            model_command="python3 -m app.runtime.llama_cpp_server",
            model_path="",
        ),
        popen_factory=lambda *a, **k: FakeProcess(),
        health_check=lambda _config: True,
    )
    runtime.validate_startup_config()

    assert runtime.health_snapshot()["fatal"] is False


def test_missing_engine_binary_is_memoized_as_fatal(monkeypatch):
    monkeypatch.delenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def missing_binary(*_args, **_kwargs):
        raise FileNotFoundError("no such file: vllm")

    runtime = ManagedRuntime(
        _runtime_config(
            service_role="summarization",
            runtime="command",
            model_command="definitely-missing-binary --serve",
            model_path="",
        ),
        popen_factory=missing_binary,
        health_check=lambda _config: True,
    )

    with pytest.raises(RuntimeError, match="no such file"):
        runtime.ensure_started()
    assert runtime.health_snapshot()["fatal"] is True


def test_model_candidates_reject_vllm_runtime(monkeypatch):
    monkeypatch.setenv("MODEL_RUNTIME", "vllm")
    monkeypatch.setenv("VLLM_MODEL_ID", "Qwen/some-model")
    monkeypatch.setenv("MODEL_CANDIDATES_JSON", '[{"model":"/models/a.gguf"}]')

    with pytest.raises(RuntimeError, match="MODEL_RUNTIME=command"):
        RuntimeConfig.from_env()


def test_model_candidates_reject_malformed_json(monkeypatch):
    monkeypatch.setenv("MODEL_RUNTIME", "command")
    monkeypatch.setenv("MODEL_COMMAND", "python -m app.runtime.llama_cpp_server")
    monkeypatch.setenv("MODEL_CANDIDATES_JSON", "{not json")

    with pytest.raises(RuntimeError, match="not valid JSON"):
        RuntimeConfig.from_env()


def _candidate_runtime(tmp_path, candidates, host_visible="1"):
    runtime = ManagedRuntime(
        _runtime_config(
            runtime="command",
            model_command="python -m app.runtime.llama_cpp_server",
            model_path="",
            service_role="vision",
            model_candidates=tuple(candidates),
        ),
        popen_factory=lambda *a, **k: FakeProcess(),
        health_check=lambda _config: True,
    )
    runtime._child_host_visible = host_visible
    return runtime


def test_candidate_selection_prefers_largest_that_fits(monkeypatch, tmp_path):
    big = tmp_path / "big.gguf"
    big.write_bytes(b"g")
    big_mmproj = tmp_path / "big-mmproj.gguf"
    big_mmproj.write_bytes(b"g")
    small = tmp_path / "small.gguf"
    small.write_bytes(b"g")

    candidates = [
        {"model": str(big), "mmproj": str(big_mmproj), "memory_gb": 21.0},
        {"model": str(small), "mmproj": "", "memory_gb": 7.0},
    ]
    runtime = _candidate_runtime(tmp_path, candidates)
    monkeypatch.setattr(
        "app.runtime.managed_openai_runtime._query_nvidia_gpus",
        lambda: [{"index": 1, "free_vram_gb": 23.5}],
    )

    env = {}
    runtime._apply_model_candidate_locked(env)

    assert env["MODEL_PATH"] == str(big)
    assert env["MODEL_MMPROJ"] == str(big_mmproj)
    assert env["MODEL_NAME"] == "big.gguf"


def test_candidate_selection_falls_through_on_vram_and_missing_files(monkeypatch, tmp_path):
    missing = tmp_path / "not-downloaded.gguf"
    small = tmp_path / "small.gguf"
    small.write_bytes(b"g")

    candidates = [
        {"model": str(missing), "mmproj": "", "memory_gb": 21.0},
        {"model": str(small), "mmproj": "", "memory_gb": 7.0},
    ]
    runtime = _candidate_runtime(tmp_path, candidates)
    monkeypatch.setattr(
        "app.runtime.managed_openai_runtime._query_nvidia_gpus",
        lambda: [{"index": 1, "free_vram_gb": 9.0}],
    )

    env = {"MODEL_MMPROJ": "stale"}
    runtime._apply_model_candidate_locked(env)

    assert env["MODEL_PATH"] == str(small)
    assert "MODEL_MMPROJ" not in env


def test_candidate_exhaustion_is_transient_not_fatal(monkeypatch, tmp_path):
    candidates = [{"model": str(tmp_path / "absent.gguf"), "mmproj": "", "memory_gb": 7.0}]
    runtime = _candidate_runtime(tmp_path, candidates)

    with pytest.raises(TransientStartError, match="not downloaded"):
        runtime._apply_model_candidate_locked({})

    # Transient failures must not poison /health.
    assert runtime.fatal_config_error() is None


def test_explicit_model_path_override_disables_candidates(monkeypatch, tmp_path):
    small = tmp_path / "small.gguf"
    small.write_bytes(b"g")
    runtime = _candidate_runtime(
        tmp_path, [{"model": str(small), "mmproj": "", "memory_gb": 7.0}]
    )

    env = {"MODEL_PATH": "/models/operator-pinned.gguf", "MODEL_MMPROJ": "/models/pin.mmproj"}
    runtime._apply_model_candidate_locked(env)

    assert env["MODEL_PATH"] == "/models/operator-pinned.gguf"
    assert env["MODEL_MMPROJ"] == "/models/pin.mmproj"


def test_candidate_exhaustion_message_contains_no_paths(monkeypatch, tmp_path):
    candidates = [
        {"model": str(tmp_path / "secret-dir" / "absent.gguf"), "mmproj": "", "memory_gb": 7.0}
    ]
    runtime = _candidate_runtime(tmp_path, candidates)

    with pytest.raises(TransientStartError) as excinfo:
        runtime._apply_model_candidate_locked({})

    message = str(excinfo.value)
    assert "absent.gguf" in message
    assert "secret-dir" not in message


def test_shutdown_after_request_spares_warming_child(monkeypatch):
    process = FakeProcess()
    runtime = ManagedRuntime(
        _runtime_config(
            runtime="command",
            model_command="python -m app.runtime.llama_cpp_server",
            model_path="/models/model.gguf",
            shutdown_after_request=True,
        ),
        popen_factory=lambda *a, **k: process,
        health_check=lambda _config: False,
    )
    monkeypatch.delenv("NVIDIA_VISION_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_SUMMARIZATION_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    # Warm-up request: child spawns, not ready, handler returns 503 and
    # finishes the request. The child must survive.
    runtime.request_started()
    assert runtime.ensure_started() is False
    runtime._process_group_id = None
    runtime.request_finished()
    assert process.terminated is False

    # Completed inference sets the served flag; only then may teardown run.
    runtime.request_started()
    runtime.note_served()
    runtime.request_finished()
    assert process.terminated is True


def test_pinned_small_gpu_enables_stage_swapping(monkeypatch):
    process = FakeProcess()

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="1, RTX 3090, 24576, 1, 24126, 0\n",
            stderr="",
        )

    monkeypatch.setattr("app.runtime.managed_openai_runtime.subprocess.run", fake_run)
    monkeypatch.setenv("NVIDIA_VISION_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    runtime = ManagedRuntime(
        _runtime_config(
            runtime="command",
            model_command="python -m app.runtime.llama_cpp_server",
            model_path="",
            service_role="vision",
            model_memory_gb=21.0,
            peer_model_memory_gb=20.0,
            hot_set_memory_gb=41.0,
        ),
        popen_factory=lambda *a, **k: process,
        health_check=lambda _config: False,
    )

    runtime.ensure_started()

    assert runtime._shutdown_after_request is False
    assert runtime._swap_idle_override_s == 30.0


def test_pinned_large_gpu_keeps_endpoints_resident(monkeypatch):
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="3, RTX PRO 6000, 97887, 1, 97000, 0\n",
            stderr="",
        )

    monkeypatch.setattr("app.runtime.managed_openai_runtime.subprocess.run", fake_run)
    monkeypatch.setenv("NVIDIA_VISION_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    runtime = ManagedRuntime(
        _runtime_config(
            runtime="command",
            model_command="python -m app.runtime.llama_cpp_server",
            model_path="",
            service_role="vision",
            hot_set_memory_gb=41.0,
        ),
        popen_factory=lambda *a, **k: FakeProcess(),
        health_check=lambda _config: False,
    )

    runtime.ensure_started()

    assert runtime._shutdown_after_request is False


def test_llama_server_runtime_accepts_candidates_only(monkeypatch):
    monkeypatch.setenv("MODEL_RUNTIME", "llama_server")
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_COMMAND", raising=False)
    monkeypatch.delenv("VLLM_MODEL_ID", raising=False)
    monkeypatch.setenv(
        "MODEL_CANDIDATES_JSON",
        '[{"model":"/models/a.gguf","mmproj":"/models/a.mmproj","memory_gb":21}]',
    )

    config = RuntimeConfig.from_env()

    assert config.runtime == "llama_server"
    assert config.model_candidates[0]["model"] == "/models/a.gguf"
    assert config.model_name == "a.gguf"


def test_build_runtime_command_applies_candidate_overrides(monkeypatch):
    monkeypatch.setattr(
        "app.runtime.managed_openai_runtime._discover_llama_server_bin",
        lambda: "/opt/cuda-llama/bin/llama-server",
    )
    config = _runtime_config(model_path="", mmproj_path="")

    command = build_runtime_command(
        config,
        model_path_override="/models/chosen.gguf",
        mmproj_override="/models/chosen.mmproj",
    )

    assert command[1:3] == ["-m", "/models/chosen.gguf"]
    mmproj_index = command.index("--mmproj")
    assert command[mmproj_index + 1] == "/models/chosen.mmproj"
