import builtins
import sys

import pytest


@pytest.fixture()
def compat(monkeypatch):
    """Fresh module state per test: reset the memo flag and sys.modules entries."""
    from app.core import torchcodec_compat

    monkeypatch.setattr(torchcodec_compat, "_checked", False)
    monkeypatch.delitem(sys.modules, "torchcodec", raising=False)
    monkeypatch.delitem(sys.modules, "torchcodec.decoders", raising=False)
    return torchcodec_compat


def _block_torchcodec_import(monkeypatch, exc):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torchcodec" or name.startswith("torchcodec."):
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_broken_torchcodec_installs_stub(compat, monkeypatch):
    _block_torchcodec_import(
        monkeypatch, RuntimeError("Could not load libtorchcodec")
    )

    assert compat.ensure_torchcodec_importable_or_stub() is False

    stub = sys.modules["torchcodec"]
    assert isinstance(stub, compat._StubModule)
    # transformers' preprocess isinstance check must be False for dict inputs.
    assert not isinstance(
        {"array": [], "sampling_rate": 16000}, stub.decoders.AudioDecoder
    )


def test_result_is_memoized_after_stub(compat, monkeypatch):
    _block_torchcodec_import(monkeypatch, RuntimeError("boom"))
    assert compat.ensure_torchcodec_importable_or_stub() is False
    # Second call must not re-import; report the stubbed state.
    assert compat.ensure_torchcodec_importable_or_stub() is False


def test_working_torchcodec_is_left_alone(compat, monkeypatch):
    fake_real = type(sys)("torchcodec")
    monkeypatch.setitem(sys.modules, "torchcodec", fake_real)

    assert compat.ensure_torchcodec_importable_or_stub() is True
    assert sys.modules["torchcodec"] is fake_real
    assert not isinstance(sys.modules["torchcodec"], compat._StubModule)
