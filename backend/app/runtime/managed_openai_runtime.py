"""Managed OpenAI-compatible runtime gateway for ROCm model servers."""

from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable


logger = logging.getLogger(__name__)

DEFAULT_HEALTH_BODY = {
    "status": "ok",
}
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _discover_llama_server_bin() -> str:
    configured = os.environ.get("LLAMA_SERVER_BIN", "").strip()
    if configured:
        return configured

    for root in ("/opt/amd-llama", "/opt/cuda-llama"):
        for candidate in Path(root).rglob("llama-server"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    raise RuntimeError(
        "Unable to locate an executable llama-server binary in /opt/amd-llama or /opt/cuda-llama."
    )


def _parse_extra_args(value: str | None) -> list[str]:
    return shlex.split(value or "")


def _split_device_tokens(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _numeric_visible_devices(value: str | None) -> tuple[int, ...]:
    tokens = _split_device_tokens(value or "")
    if not tokens:
        return ()
    try:
        return tuple(int(token) for token in tokens)
    except ValueError:
        return ()


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _translate_device_index_value(value: str, visible_devices: tuple[int, ...]) -> str:
    if not visible_devices:
        return value

    prefix = ""
    raw_index = value.strip()
    lowered = raw_index.lower()
    if lowered.startswith("cuda:"):
        prefix = raw_index[: raw_index.index(":") + 1]
        raw_index = raw_index[len(prefix):]

    try:
        host_index = int(raw_index)
    except ValueError:
        return value

    if host_index in visible_devices:
        return f"{prefix}{visible_devices.index(host_index)}"

    # Already-local ordinals are left alone for operators who intentionally pass them.
    if 0 <= host_index < len(visible_devices):
        return value
    return value


def _translate_tensor_split_value(value: str, visible_devices: tuple[int, ...]) -> str:
    if not visible_devices:
        return value

    tokens = _split_device_tokens(value)
    if not tokens:
        return value

    keyed_values: dict[int, str] = {}
    for token in tokens:
        separator = ":" if ":" in token else "=" if "=" in token else ""
        if not separator:
            keyed_values = {}
            break
        key, ratio = token.split(separator, 1)
        try:
            keyed_values[int(key.strip())] = ratio.strip()
        except ValueError:
            return value
    if keyed_values:
        return ",".join(keyed_values.get(host_index, "0") for host_index in visible_devices)

    if len(tokens) > max(visible_devices) and all(_is_number(token) for token in tokens):
        return ",".join(tokens[host_index] for host_index in visible_devices)
    return value


def _translate_engine_device_args(args: list[str], cuda_visible_devices: str | None) -> list[str]:
    visible_devices = _numeric_visible_devices(cuda_visible_devices)
    if not visible_devices:
        return list(args)

    device_index_options = {
        "--main-gpu",
        "--main_gpu",
        "-mg",
        "--gpu",
        "--gpu-id",
        "--gpu_id",
        "--device-id",
        "--device_id",
        "--cuda-device",
        "--cuda_device",
        "--device",
    }
    tensor_split_options = {"--tensor-split", "--tensor_split", "-ts"}
    translated: list[str] = []
    index = 0
    while index < len(args):
        current = args[index]
        if "=" in current:
            option, value = current.split("=", 1)
            if option in device_index_options:
                translated.append(f"{option}={_translate_device_index_value(value, visible_devices)}")
                index += 1
                continue
            if option in tensor_split_options:
                translated.append(f"{option}={_translate_tensor_split_value(value, visible_devices)}")
                index += 1
                continue

        if current in device_index_options and index + 1 < len(args):
            translated.extend([current, _translate_device_index_value(args[index + 1], visible_devices)])
            index += 2
            continue
        if current in tensor_split_options and index + 1 < len(args):
            translated.extend([current, _translate_tensor_split_value(args[index + 1], visible_devices)])
            index += 2
            continue

        translated.append(current)
        index += 1
    return translated


def _normalize_vllm_args(args: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(args):
        current = args[index]
        if current == "--limit-mm-per-prompt" and index + 1 < len(args):
            raw_values: list[str] = []
            probe = index + 1
            while probe < len(args) and not args[probe].startswith("-"):
                raw_values.append(args[probe])
                probe += 1

            if raw_values and all("=" in item for item in raw_values):
                limits: dict[str, object] = {}
                for item in raw_values:
                    key, value = item.split("=", 1)
                    parsed_value: object = value
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                    limits[key] = parsed_value
                normalized.extend([current, json.dumps(limits, separators=(",", ":"))])
                index = probe
                continue

        normalized.append(current)
        index += 1
    return normalized


def _parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _explicit_visible_device_keys(service_role: str) -> tuple[str, ...]:
    role = service_role.strip().lower()
    if role == "vision":
        return ("MODEL_VISIBLE_DEVICES", "NVIDIA_VISION_VISIBLE_DEVICES")
    if role == "summarization":
        return ("MODEL_VISIBLE_DEVICES", "NVIDIA_SUMMARIZATION_VISIBLE_DEVICES")
    return ("MODEL_VISIBLE_DEVICES", "NVIDIA_VISION_VISIBLE_DEVICES", "NVIDIA_SUMMARIZATION_VISIBLE_DEVICES")


_PIN_ENV_KEYS = (
    "MODEL_VISIBLE_DEVICES",
    "NVIDIA_VISION_VISIBLE_DEVICES",
    "NVIDIA_SUMMARIZATION_VISIBLE_DEVICES",
)


def _resolve_pinned_child_devices(
    pin: str,
    parent_visible: str,
    *,
    role: str,
) -> tuple[str, tuple[int, ...]]:
    """Resolve host-index GPU pins into the child's ``CUDA_VISIBLE_DEVICES``.

    ``CUDA_VISIBLE_DEVICES`` does not compose across processes: when the child
    exports its own value, CUDA re-enumerates against the devices the container
    exposes (the physical/nvidia-smi namespace), not against the parent's own
    narrowed list. Pins are therefore exported VERBATIM as physical ids —
    translating them to parent-relative ordinals selects the wrong card
    (confirmed live: parent ``1,4,6`` + pin ``4`` exported as ``1`` landed on
    physical GPU 1).

    Returns ``(child_cuda_visible_devices, host_device_ids)``; engine device
    arguments are translated later to ordinals within that child list. A parent
    ``CUDA_VISIBLE_DEVICES`` acts as a deployment allowlist: pins outside it are
    dropped with a warning, and a pin set that resolves to nothing fails fast
    rather than letting the engine grab device 0.
    """
    role_label = role or "default"
    pin_tokens = _split_device_tokens(pin)
    parent_tokens = _split_device_tokens(parent_visible)

    child_tokens: list[str] = []
    for token in pin_tokens:
        if parent_tokens and token not in parent_tokens:
            logger.warning(
                "Ignoring GPU pin %r for role %r: it is not in the deployment's visible "
                "CUDA devices allowlist %s (CUDA_VISIBLE_DEVICES).",
                token,
                role_label,
                list(parent_tokens),
            )
            continue
        child_tokens.append(token)

    if not child_tokens:
        raise RuntimeError(
            f"None of the pinned GPUs {list(pin_tokens)} for role {role_label!r} are "
            f"usable within the visible CUDA devices {list(parent_tokens)}. Set the pin "
            f"to a physical GPU index (as reported by nvidia-smi) that is included in "
            f"CUDA_VISIBLE_DEVICES."
        )
    child_visible = ",".join(child_tokens)
    return child_visible, _numeric_visible_devices(child_visible)


def _query_nvidia_gpus() -> list[dict[str, object]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    if result.returncode != 0:
        return []

    gpus: list[dict[str, object]] = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_vram_gb": float(parts[2]) / 1024.0,
                    "used_vram_gb": float(parts[3]) / 1024.0,
                    "free_vram_gb": float(parts[4]) / 1024.0,
                    "utilization_gpu": float(parts[5]) if len(parts) > 5 else 0.0,
                }
            )
        except ValueError:
            continue
    return sorted(gpus, key=lambda item: _device_preference_key(item))


def _used_capacity_gb(gpu: dict[str, object]) -> float:
    return float(gpu.get("used_vram_gb", 0.0) or 0.0)


def _utilization_percent(gpu: dict[str, object]) -> float:
    return float(gpu.get("utilization_gpu", 0.0) or 0.0)


def _occupancy_ratio(gpu: dict[str, object]) -> float:
    total = float(gpu.get("total_vram_gb", 0.0) or 0.0)
    if total <= 0:
        return 1.0
    return min(max(_used_capacity_gb(gpu) / total, 0.0), 1.0)


def _device_preference_key(
    gpu: dict[str, object],
    *,
    requirement_gb: float = 0.0,
) -> tuple[float, float, float, float]:
    free_vram_gb = float(gpu.get("free_vram_gb", 0.0) or 0.0)
    can_fit = 0.0 if free_vram_gb >= max(requirement_gb, 0.0) else 1.0
    return (
        can_fit,
        _occupancy_ratio(gpu),
        _utilization_percent(gpu),
        -free_vram_gb,
    )


class TransientStartError(RuntimeError):
    """Startup blocker that can heal without a container restart.

    Example: no candidate model file downloaded yet, or no candidate fits
    the currently free VRAM. Never memoized as fatal configuration.
    """


def _parse_swap_idle_timeout(raw: str | None) -> float:
    value = (raw or "").strip()
    if not value:
        return 30.0
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(
            f"SWAP_IDLE_TIMEOUT_SECONDS must be a positive number, got {value!r}."
        ) from exc
    if not (parsed > 0) or parsed != parsed or parsed == float("inf"):
        raise RuntimeError(
            f"SWAP_IDLE_TIMEOUT_SECONDS must be a positive finite number, got {value!r}."
        )
    return parsed


def _parse_model_candidates(raw: str | None) -> tuple[dict, ...]:
    value = (raw or "").strip()
    if not value:
        return ()
    try:
        payload = json.loads(value)
    except ValueError as exc:
        raise RuntimeError(f"MODEL_CANDIDATES_JSON is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise RuntimeError("MODEL_CANDIDATES_JSON must be a JSON array.")
    candidates: list[dict] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict) or not str(entry.get("model", "")).strip():
            raise RuntimeError(
                f"MODEL_CANDIDATES_JSON entry {index} must be an object with a 'model' path."
            )
        candidates.append(
            {
                "model": str(entry["model"]).strip(),
                "mmproj": str(entry.get("mmproj", "")).strip(),
                "memory_gb": float(entry.get("memory_gb", 0) or 0),
            }
        )
    return tuple(candidates)


@dataclass(frozen=True)
class RuntimeConfig:
    runtime: str
    model_command: str
    model_path: str
    model_name: str
    vllm_model_id: str
    service_role: str
    auto_nvidia_gpu_selection: bool
    model_memory_gb: float
    peer_model_memory_gb: float
    hot_set_memory_gb: float
    shutdown_after_request: bool
    host: str
    public_port: int
    upstream_port: int
    context_size: int
    gpu_layers: int
    mmproj_path: str
    idle_timeout_seconds: int
    startup_timeout_seconds: int
    proxy_timeout_seconds: int
    llama_extra_args: list[str]
    vllm_extra_args: list[str]
    model_candidates: tuple = ()
    swap_idle_timeout_s: float = 30.0

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        runtime = os.environ.get("MODEL_RUNTIME", "llama_server").strip() or "llama_server"
        model_command = os.environ.get("MODEL_COMMAND", "").strip()
        model_path = os.environ.get("MODEL_PATH", "").strip()
        vllm_model_id = os.environ.get("VLLM_MODEL_ID", "").strip()
        model_candidates = _parse_model_candidates(os.environ.get("MODEL_CANDIDATES_JSON"))
        if model_candidates and runtime not in {"command", "llama_server"}:
            raise RuntimeError(
                "MODEL_CANDIDATES_JSON requires MODEL_RUNTIME=command or "
                "llama_server."
            )
        effective_model_source = model_path or vllm_model_id or model_command
        if not effective_model_source and model_candidates:
            effective_model_source = model_candidates[0]["model"]
        if not effective_model_source:
            raise RuntimeError("MODEL_PATH is required.")

        model_name = os.environ.get("MODEL_NAME", "").strip() or Path(effective_model_source).name

        return cls(
            runtime=runtime,
            model_command=model_command,
            model_path=model_path,
            model_name=model_name,
            vllm_model_id=vllm_model_id,
            service_role=os.environ.get("SERVICE_ROLE", "").strip().lower(),
            auto_nvidia_gpu_selection=_parse_bool(os.environ.get("AUTO_NVIDIA_GPU_SELECTION")),
            model_memory_gb=float(os.environ.get("MODEL_MEMORY_GB", "0") or 0),
            peer_model_memory_gb=float(os.environ.get("PEER_MODEL_MEMORY_GB", "0") or 0),
            hot_set_memory_gb=float(os.environ.get("HOT_SET_MEMORY_GB", "0") or 0),
            shutdown_after_request=_parse_bool(os.environ.get("SHUTDOWN_AFTER_REQUEST")),
            host=os.environ.get("LISTEN_HOST", "0.0.0.0"),
            public_port=int(os.environ.get("PORT", "8080")),
            upstream_port=int(os.environ.get("UPSTREAM_PORT", "18080")),
            context_size=int(os.environ.get("MODEL_CONTEXT_SIZE", "8192")),
            gpu_layers=int(os.environ.get("MODEL_GPU_LAYERS", "999")),
            mmproj_path=os.environ.get("MODEL_MMPROJ", "").strip(),
            idle_timeout_seconds=int(os.environ.get("IDLE_TIMEOUT_SECONDS", "120")),
            startup_timeout_seconds=int(os.environ.get("STARTUP_TIMEOUT_SECONDS", "600")),
            proxy_timeout_seconds=int(os.environ.get("PROXY_TIMEOUT_SECONDS", "600")),
            llama_extra_args=_parse_extra_args(os.environ.get("LLAMA_SERVER_EXTRA_ARGS")),
            vllm_extra_args=_parse_extra_args(os.environ.get("VLLM_EXTRA_ARGS")),
            model_candidates=model_candidates,
            swap_idle_timeout_s=_parse_swap_idle_timeout(
                os.environ.get("SWAP_IDLE_TIMEOUT_SECONDS")
            ),
        )


def build_runtime_command(
    config: RuntimeConfig,
    *,
    cuda_visible_devices: str | None = None,
    model_path_override: str | None = None,
    mmproj_override: str | None = None,
) -> list[str]:
    model_path = (model_path_override or "").strip() or config.model_path
    # None = no override supplied; empty string = candidate explicitly cleared
    # the projector (never resurrect a stale configured mmproj).
    if mmproj_override is None:
        mmproj_path = config.mmproj_path
    else:
        mmproj_path = mmproj_override.strip()
    if config.runtime == "llama_server":
        command = [
            _discover_llama_server_bin(),
            "-m",
            model_path,
            "-c",
            str(config.context_size),
            "-ngl",
            str(config.gpu_layers),
            "-np",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            str(config.upstream_port),
        ]
        if mmproj_path:
            command.extend(["--mmproj", mmproj_path])
        command.extend(_translate_engine_device_args(config.llama_extra_args, cuda_visible_devices))
        return command

    if config.runtime == "vllm":
        model_source = config.vllm_model_id or config.model_path
        command = [
            "vllm",
            "serve",
            model_source,
            "--host",
            "127.0.0.1",
            "--port",
            str(config.upstream_port),
            "--served-model-name",
            config.model_name,
            "--max-model-len",
            str(config.context_size),
            "--tensor-parallel-size",
            "1",
        ]
        command.extend(_translate_engine_device_args(_normalize_vllm_args(config.vllm_extra_args), cuda_visible_devices))
        return command

    if config.runtime == "command":
        if not config.model_command:
            raise RuntimeError("MODEL_COMMAND is required when MODEL_RUNTIME=command.")
        return _translate_engine_device_args(shlex.split(config.model_command), cuda_visible_devices)

    raise RuntimeError(f"Unsupported MODEL_RUNTIME: {config.runtime}")


class ManagedRuntime:
    def __init__(
        self,
        config: RuntimeConfig,
        *,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        health_check: Callable[[RuntimeConfig], bool] | None = None,
    ) -> None:
        self.config = config
        self._popen_factory = popen_factory
        self._clock = clock
        self._sleep = sleeper
        self._health_check = health_check or self._default_health_check
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._process_group_id: int | None = None
        self._child_host_visible: str | None = None
        self._last_request_ts = 0.0
        self._last_start_ts = 0.0
        self._active_requests = 0
        self._shutdown_after_request = config.shutdown_after_request
        self._fatal_config_error: str | None = None
        self._served_since_start = False
        self._swap_idle_override_s: float | None = None

    def fatal_config_error(self) -> str | None:
        with self._lock:
            return self._fatal_config_error

    def health_snapshot(self) -> dict:
        """Fatal state and child status read under one lock acquisition."""
        with self._lock:
            running = self._child_running_locked()
            return {
                "fatal": self._fatal_config_error is not None,
                "child_running": running,
                "child_ready": running and self._health_check(self.config),
            }

    def validate_startup_config(self) -> None:
        """Fail fast on immutable configuration errors.

        Explicit device pins and the runtime command shape cannot change for
        the container's lifetime, so an invalid value must flip /health
        immediately instead of waiting for the first proxied request to
        discover it. Auto GPU selection depends on live occupancy and is
        deliberately not validated here.
        """
        with self._lock:
            try:
                has_pin = any(
                    os.environ.get(key, "").strip()
                    for key in _explicit_visible_device_keys(self.config.service_role)
                )
                if has_pin:
                    self._build_child_env_locked()
                build_runtime_command(
                    self.config, cuda_visible_devices=self._child_host_visible
                )
            except (RuntimeError, OSError) as exc:
                self._fatal_config_error = str(exc)
                logger.error("Fatal endpoint configuration error at startup: %s", exc)

    def note_activity(self) -> None:
        with self._lock:
            self._last_request_ts = self._clock()

    def note_served(self) -> None:
        with self._lock:
            self._served_since_start = True

    def request_started(self) -> None:
        with self._lock:
            self._last_request_ts = self._clock()
            self._active_requests += 1

    def request_finished(self) -> None:
        with self._lock:
            self._last_request_ts = self._clock()
            if self._active_requests > 0:
                self._active_requests -= 1
            if (
                self._active_requests == 0
                and self._shutdown_after_request
                and self._served_since_start
                and self._child_running_locked()
            ):
                # Only tear down after a completed inference; a warm-up 503
                # must never kill the still-loading child (retry loop of death
                # on constrained single-GPU hosts).
                self._terminate_locked()

    def child_running(self) -> bool:
        with self._lock:
            return self._child_running_locked()

    def child_ready(self) -> bool:
        with self._lock:
            return self._child_running_locked() and self._health_check(self.config)

    def ensure_started(self) -> bool:
        with self._lock:
            self._last_request_ts = self._clock()
            if self._fatal_config_error:
                raise RuntimeError(self._fatal_config_error)
            if self._child_running_locked():
                return self._health_check(self.config)

            try:
                env = self._build_child_env_locked()
                self._apply_model_candidate_locked(env)
                command = build_runtime_command(
                    self.config,
                    cuda_visible_devices=self._child_host_visible,
                    model_path_override=env.get("MODEL_PATH"),
                    mmproj_override=env.get("MODEL_MMPROJ"),
                )
                self._process = self._popen_factory(command, env=env, start_new_session=True)
            except TransientStartError:
                raise
            except (RuntimeError, OSError) as exc:
                # Device pins, runtime command shape, and missing/non-executable
                # engine binaries cannot heal without a container restart;
                # memoize so /health reports the failure instead of the wrapper
                # sitting healthy with no usable child.
                self._fatal_config_error = str(exc)
                logger.error("Fatal endpoint configuration error: %s", exc)
                raise RuntimeError(str(exc)) from exc
            self._process_group_id = getattr(self._process, "pid", None)
            self._served_since_start = False
            self._last_start_ts = self._last_request_ts
            return False

    def maybe_stop_idle_process(self) -> bool:
        timeout = (
            self._swap_idle_override_s
            if self._swap_idle_override_s is not None
            else self.config.idle_timeout_seconds
        )
        if timeout <= 0:
            return False

        with self._lock:
            if not self._child_running_locked():
                return False
            if self._active_requests > 0:
                return False
            now = self._clock()
            if (now - self._last_request_ts) < timeout:
                return False
            self._terminate_locked()
            return True

    def _child_running_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _build_child_env_locked(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if self.config.runtime == "command":
            env["PORT"] = str(self.config.upstream_port)
            env["LISTEN_HOST"] = "127.0.0.1"

        parent_visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
        self._shutdown_after_request = self.config.shutdown_after_request

        pin = ""
        for key in _explicit_visible_device_keys(self.config.service_role):
            value = env.get(key, "").strip()
            if value:
                pin = value
                break

        # An explicit role pin (expressed as HOST indices) narrows visibility,
        # translated against any parent narrowing already in effect.
        if pin:
            child_visible, host_ids = _resolve_pinned_child_devices(
                pin, parent_visible, role=self.config.service_role
            )
            self._maybe_enable_swap_idle_locked(host_ids)
            return self._finalize_child_env_locked(env, child_visible, host_ids)

        # Auto-pick from live nvidia-smi data. nvidia-smi reports HOST indices even
        # under a narrowed CUDA_VISIBLE_DEVICES, so restrict candidates to the
        # visible set and translate the chosen index the same way pins are.
        if self.config.auto_nvidia_gpu_selection:
            gpus = _query_nvidia_gpus()
            if parent_visible:
                visible_set = _split_device_tokens(parent_visible)
                gpus = [gpu for gpu in gpus if str(int(gpu["index"])) in visible_set]
            if gpus:
                visible_devices, shutdown_after_request = _select_nvidia_visible_devices(
                    self.config, gpus
                )
                if shutdown_after_request and not self.config.shutdown_after_request:
                    self._swap_idle_override_s = self.config.swap_idle_timeout_s
                else:
                    self._shutdown_after_request = shutdown_after_request
                if visible_devices:
                    child_visible, host_ids = _resolve_pinned_child_devices(
                        ",".join(visible_devices), parent_visible, role=self.config.service_role
                    )
                    return self._finalize_child_env_locked(env, child_visible, host_ids)

        # No pin and no auto-selection: honor any pre-narrowed parent visibility as-is.
        self._child_host_visible = parent_visible or None
        return env

    def _finalize_child_env_locked(
        self,
        env: dict[str, str],
        child_visible: str,
        host_ids: tuple[int, ...],
    ) -> dict[str, str]:
        env["CUDA_VISIBLE_DEVICES"] = child_visible
        # This gateway is the sole authority on device selection; drop the raw
        # host-index pins so a chained child (e.g. llama_cpp_server) does not
        # re-translate the already-narrowed visibility a second time.
        for key in _PIN_ENV_KEYS:
            env.pop(key, None)
        self._child_host_visible = ",".join(str(index) for index in host_ids) or child_visible
        return env

    def _maybe_enable_swap_idle_locked(self, host_ids: tuple[int, ...]) -> None:
        """Enable stage swapping when the target GPUs cannot hold the hot set.

        Swap mode uses a short idle linger rather than per-request teardown:
        multi-request stages (vision analyzes many frames) keep the child warm
        between calls instead of reloading tens of GB per request, and the
        card frees a few seconds after the stage ends so the peer endpoint
        can load. Explicit SHUTDOWN_AFTER_REQUEST=true keeps its old
        per-request behavior.
        """
        if self._shutdown_after_request or self._swap_idle_override_s is not None:
            return
        if not host_ids:
            return
        hot_set_gb = self.config.hot_set_memory_gb
        if hot_set_gb <= 0:
            return
        try:
            gpus = _query_nvidia_gpus()
            capacity_gb = sum(
                float(gpu["total_vram_gb"])
                for gpu in gpus
                if int(gpu["index"]) in set(host_ids)
            )
        except Exception:
            return
        if capacity_gb and hot_set_gb > capacity_gb:
            self._swap_idle_override_s = self.config.swap_idle_timeout_s
            logger.info(
                "GPUs %s hold %.1f GB but the endpoint hot set needs %.1f GB; "
                "swap mode active with a %.0fs idle linger.",
                list(host_ids),
                capacity_gb,
                hot_set_gb,
                self._swap_idle_override_s,
            )

    def _selected_devices_memory_gib(self) -> tuple[float, float] | None:
        """(min free, min total) GiB across the selected devices, or None."""
        visible = (self._child_host_visible or "").strip()
        if not visible:
            return None
        try:
            wanted = {int(token) for token in visible.split(",") if token.strip()}
            gpus = _query_nvidia_gpus()
            rows = [gpu for gpu in gpus if int(gpu["index"]) in wanted]
            if not rows:
                return None
            return (
                min(float(gpu["free_vram_gb"]) for gpu in rows),
                min(float(gpu["total_vram_gb"]) for gpu in rows),
            )
        except Exception:
            return None

    def _apply_model_candidate_locked(self, env: dict[str, str]) -> None:
        """Pick the largest downloaded candidate that fits the selected GPU.

        Core directive: best model for the platform. Candidates are ordered
        best-first in MODEL_CANDIDATES_JSON; a candidate qualifies when its
        file exists and (when telemetry is available) its memory need plus
        headroom fits the selected device's free VRAM. Failure here is
        transient — downloading a model or freeing VRAM heals it without a
        container restart.
        """
        candidates = self.config.model_candidates
        if not candidates:
            return
        if (env.get("MODEL_PATH") or "").strip():
            # Explicit operator override always wins over candidate selection.
            return
        memory = self._selected_devices_memory_gib()
        free_gib = memory[0] if memory else None
        total_gib = memory[1] if memory else None
        skipped: list[str] = []
        waiting_for_memory = False
        for candidate in candidates:
            model_path = candidate["model"]
            if not os.path.isfile(model_path):
                skipped.append(f"{model_path} (not downloaded)")
                continue
            mmproj = candidate["mmproj"]
            if mmproj and not os.path.isfile(mmproj):
                skipped.append(f"{model_path} (missing mmproj {mmproj})")
                continue
            need_gib = candidate["memory_gb"]
            if free_gib is not None and need_gib and need_gib + 1.0 > free_gib:
                if total_gib is None or need_gib + 1.0 <= total_gib:
                    # Fits the card, just not right now (peer endpoint still
                    # lingering). Wait for the swap instead of silently
                    # downgrading to a smaller model — best bang wins.
                    waiting_for_memory = True
                    skipped.append(
                        f"{model_path} (needs ~{need_gib:.1f} GiB, {free_gib:.1f} GiB free; waiting)"
                    )
                    break
                skipped.append(
                    f"{model_path} (needs ~{need_gib:.1f} GiB, card holds {total_gib:.1f} GiB)"
                )
                continue
            env["MODEL_PATH"] = model_path
            if not (env.get("MODEL_NAME") or "").strip():
                env["MODEL_NAME"] = Path(model_path).name
            # Empty string is the explicit "no projector" sentinel; a missing
            # key would let a stale configured mmproj resurface in the argv.
            env["MODEL_MMPROJ"] = mmproj or ""
            logger.info("Selected model candidate %s for %s.", model_path, self.config.service_role or "endpoint")
            return
        # Full diagnostics (paths, free VRAM) go to logs; the HTTP surface gets
        # model identifiers and guidance only.
        logger.warning("No usable model candidate: %s", "; ".join(skipped))
        names = ", ".join(Path(candidate["model"]).name for candidate in candidates)
        if waiting_for_memory:
            # "Loading model" is the canonical retryable token the pipeline
            # clients poll on; peer swap-out frees the memory shortly.
            raise TransientStartError(
                f"Loading model: waiting for GPU memory to serve {names}."
            )
        raise TransientStartError(
            f"No approved model is ready ({names}): each is either not "
            "downloaded or does not fit this GPU. Download one via "
            "Settings → Model Downloads."
        )

    def _terminate_locked(self) -> None:
        if not self._process:
            return
        process_group_id = self._process_group_id
        if process_group_id is not None:
            try:
                os.killpg(process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if process_group_id is not None:
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                self._process.kill()
            self._process.wait(timeout=30)
        finally:
            self._process = None
            self._process_group_id = None

    def _default_health_check(self, config: RuntimeConfig) -> bool:
        urls = [f"http://127.0.0.1:{config.upstream_port}/health"]
        if config.runtime == "command":
            urls.append(f"http://127.0.0.1:{config.upstream_port}/v1/models")

        for url in urls:
            request = urllib.request.Request(url, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=1.0) as response:
                    if response.status < 500:
                        return True
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                return False
            except (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout):
                return False
        return False


def _select_nvidia_visible_devices(
    config: RuntimeConfig,
    gpus: list[dict[str, object]],
) -> tuple[tuple[str, ...], bool]:
    if not gpus:
        return (), config.shutdown_after_request

    own_memory = max(float(config.model_memory_gb or 0.0), 0.0)
    peer_memory = max(float(config.peer_model_memory_gb or 0.0), 0.0)
    hot_set = max(float(config.hot_set_memory_gb or 0.0), own_memory + peer_memory)
    sorted_gpus = sorted(gpus, key=lambda gpu: _device_preference_key(gpu, requirement_gb=own_memory))
    best = sorted_gpus[0]
    best_free = float(best.get("free_vram_gb", 0.0) or 0.0)
    selected = (str(int(best["index"])),)
    shutdown_after_request = config.shutdown_after_request

    if len(sorted_gpus) == 1:
        if hot_set > 0 and best_free < hot_set:
            shutdown_after_request = True
        return selected, shutdown_after_request

    if config.service_role == "summarization":
        if hot_set > 0 and best_free >= hot_set:
            return selected, shutdown_after_request
        alternate = next(
            (
                gpu for gpu in sorted_gpus[1:]
                if float(gpu.get("free_vram_gb", 0.0) or 0.0) >= own_memory
            ),
            None,
        )
        if alternate is not None:
            return (str(int(alternate["index"])),), shutdown_after_request

    return selected, shutdown_after_request


class RuntimeProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    runtime_manager: ManagedRuntime

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def log_message(self, *_args) -> None:  # pragma: no cover - noisy stdlib override
        return

    def _handle(self) -> None:
        if self.path == "/health":
            snapshot = self.runtime_manager.health_snapshot()
            body = {
                **DEFAULT_HEALTH_BODY,
                "runtime": self.runtime_manager.config.runtime,
                "child_running": snapshot["child_running"],
                "child_ready": snapshot["child_ready"],
            }
            if snapshot["fatal"]:
                # Details (device indices, allowlists) stay in the logs; this
                # gateway may be reachable beyond localhost.
                body["status"] = "fatal"
                body["detail"] = "Endpoint configuration is invalid; see container logs."
            self._write_json(503 if snapshot["fatal"] else 200, body)
            return

        self.runtime_manager.note_activity()
        self.runtime_manager.request_started()
        try:
            try:
                ready = self.runtime_manager.ensure_started()
            except TransientStartError as exc:
                # Healable without restart (e.g. model not downloaded yet);
                # the message is actionable and contains no device topology.
                self._write_json(
                    503,
                    {
                        "error": {
                            "message": str(exc),
                            "type": "unavailable_error",
                            "code": 503,
                        }
                    },
                )
                return
            except RuntimeError:
                self._write_json(
                    503,
                    {
                        "error": {
                            "message": (
                                "Endpoint configuration is invalid; see container logs."
                            ),
                            "type": "unavailable_error",
                            "code": 503,
                        }
                    },
                )
                return
            if not ready:
                self._write_json(
                    503,
                    {
                        "error": {
                            "message": "Loading model",
                            "type": "unavailable_error",
                            "code": 503,
                        }
                    },
                )
                return

            self._proxy_to_child()
            # Teardown-after-request may only follow a completed inference;
            # warm-up 503s above never set this.
            self.runtime_manager.note_served()
        finally:
            self.runtime_manager.request_finished()

    def _proxy_to_child(self) -> None:
        body = b""
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length:
            body = self.rfile.read(content_length)

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        headers["Host"] = f"127.0.0.1:{self.runtime_manager.config.upstream_port}"

        url = f"http://127.0.0.1:{self.runtime_manager.config.upstream_port}{self.path}"
        request = urllib.request.Request(url, data=body if body else None, headers=headers, method=self.command)

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.runtime_manager.config.proxy_timeout_seconds,
            ) as response:
                payload = response.read()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() in HOP_BY_HOP_HEADERS or key.lower() == "content-length":
                        continue
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            for key, value in exc.headers.items():
                if key.lower() in HOP_BY_HOP_HEADERS or key.lower() == "content-length":
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def _write_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _idle_monitor(runtime_manager: ManagedRuntime, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        runtime_manager.maybe_stop_idle_process()
        stop_event.wait(1.0)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("RUNTIME_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = RuntimeConfig.from_env()
    runtime_manager = ManagedRuntime(config)
    runtime_manager.validate_startup_config()
    stop_event = threading.Event()

    RuntimeProxyHandler.runtime_manager = runtime_manager

    monitor = threading.Thread(
        target=_idle_monitor,
        args=(runtime_manager, stop_event),
        daemon=True,
    )
    monitor.start()

    server = ThreadingHTTPServer((config.host, config.public_port), RuntimeProxyHandler)
    try:
        server.serve_forever()
    finally:  # pragma: no cover - shutdown path
        stop_event.set()
        monitor.join(timeout=2.0)
        runtime_manager.maybe_stop_idle_process()


if __name__ == "__main__":
    main()
