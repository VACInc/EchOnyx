from types import SimpleNamespace

from app.config import GPUBackend, HardwareProfile
from app.core import diarization as diarization_module


def test_should_retry_on_cpu_after_gpu_error_for_non_strict_runtime(monkeypatch):
    settings = SimpleNamespace(
        hardware_profile=HardwareProfile.CPU_ONLY,
        gpu_backend=GPUBackend.ROCM,
    )
    monkeypatch.setattr(diarization_module, "get_settings", lambda: settings)

    assert diarization_module._should_retry_on_cpu_after_gpu_error(RuntimeError("MIOpen failure"))


def test_should_not_retry_on_cpu_after_gpu_error_for_strix_halo(monkeypatch):
    settings = SimpleNamespace(
        hardware_profile=HardwareProfile.STRIX_HALO,
        gpu_backend=GPUBackend.ROCM,
    )
    monkeypatch.setattr(diarization_module, "get_settings", lambda: settings)

    assert not diarization_module._should_retry_on_cpu_after_gpu_error(RuntimeError("HIP kernel failure"))


def test_run_diarization_pipeline_uses_eval_and_inference_mode():
    calls: list[str] = []

    class DummyContext:
        def __enter__(self):
            calls.append("enter")
            return None

        def __exit__(self, exc_type, exc, tb):
            calls.append("exit")
            return False

    class DummyPipeline:
        def eval(self):
            calls.append("eval")

        def __call__(self, inputs, **params):
            calls.append("call")
            return {"inputs": inputs, "params": params}

    class DummyTorch:
        @staticmethod
        def inference_mode():
            return DummyContext()

    result = diarization_module._run_diarization_pipeline(
        DummyPipeline(),
        waveform="waveform",
        sample_rate=16000,
        params={"num_speakers": 2},
        torch_module=DummyTorch(),
    )

    assert calls == ["eval", "enter", "call", "exit"]
    assert result["inputs"]["waveform"] == "waveform"
    assert result["params"]["num_speakers"] == 2
