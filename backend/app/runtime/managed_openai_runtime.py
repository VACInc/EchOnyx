"""Managed OpenAI-compatible runtime gateway for ROCm model servers."""

from __future__ import annotations

import json
import os
import shlex
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

    for candidate in Path("/opt/amd-llama").rglob("llama-server"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError("Unable to locate an executable llama-server binary in /opt/amd-llama.")


def _parse_extra_args(value: str | None) -> list[str]:
    return shlex.split(value or "")


@dataclass(frozen=True)
class RuntimeConfig:
    runtime: str
    model_path: str
    model_name: str
    vllm_model_id: str
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

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        runtime = os.environ.get("MODEL_RUNTIME", "llama_server").strip() or "llama_server"
        model_path = os.environ.get("MODEL_PATH", "").strip()
        vllm_model_id = os.environ.get("VLLM_MODEL_ID", "").strip()
        effective_model_source = model_path or vllm_model_id
        if not effective_model_source:
            raise RuntimeError("MODEL_PATH is required.")

        model_name = os.environ.get("MODEL_NAME", "").strip() or Path(effective_model_source).name

        return cls(
            runtime=runtime,
            model_path=model_path,
            model_name=model_name,
            vllm_model_id=vllm_model_id,
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
        )


def build_runtime_command(config: RuntimeConfig) -> list[str]:
    if config.runtime == "llama_server":
        command = [
            _discover_llama_server_bin(),
            "-m",
            config.model_path,
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
        if config.mmproj_path:
            command.extend(["--mmproj", config.mmproj_path])
        command.extend(config.llama_extra_args)
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
        command.extend(config.vllm_extra_args)
        return command

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
        self._last_request_ts = 0.0
        self._last_start_ts = 0.0
        self._active_requests = 0

    def note_activity(self) -> None:
        with self._lock:
            self._last_request_ts = self._clock()

    def request_started(self) -> None:
        with self._lock:
            self._last_request_ts = self._clock()
            self._active_requests += 1

    def request_finished(self) -> None:
        with self._lock:
            self._last_request_ts = self._clock()
            if self._active_requests > 0:
                self._active_requests -= 1

    def child_running(self) -> bool:
        with self._lock:
            return self._child_running_locked()

    def child_ready(self) -> bool:
        with self._lock:
            return self._child_running_locked() and self._health_check(self.config)

    def ensure_started(self) -> bool:
        with self._lock:
            self._last_request_ts = self._clock()
            if self._child_running_locked():
                return self._health_check(self.config)

            command = build_runtime_command(self.config)
            env = os.environ.copy()
            self._process = self._popen_factory(command, env=env)
            self._last_start_ts = self._last_request_ts
            return False

    def maybe_stop_idle_process(self) -> bool:
        timeout = self.config.idle_timeout_seconds
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

    def _terminate_locked(self) -> None:
        if not self._process:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=30)
        finally:
            self._process = None

    def _default_health_check(self, config: RuntimeConfig) -> bool:
        url = f"http://127.0.0.1:{config.upstream_port}/health"
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=1.0) as response:
                return response.status < 500
        except (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout):
            return False


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
            self._write_json(
                200,
                {
                    **DEFAULT_HEALTH_BODY,
                    "runtime": self.runtime_manager.config.runtime,
                    "child_running": self.runtime_manager.child_running(),
                    "child_ready": self.runtime_manager.child_ready(),
                },
            )
            return

        self.runtime_manager.note_activity()
        self.runtime_manager.request_started()
        try:
            ready = self.runtime_manager.ensure_started()
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
    config = RuntimeConfig.from_env()
    runtime_manager = ManagedRuntime(config)
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
