"""Guard against a torchcodec install that is present but unimportable.

transformers gates its torchcodec fast path on package *metadata* only
(`is_torchcodec_available`), then unconditionally runs `import torchcodec`
inside ASR preprocess — before checking the input type. A torchcodec wheel
built for a different accelerator stack (for example the CUDA-linked wheel
inside the ROCm image, which needs libnvrtc) passes the metadata check but
raises RuntimeError at import time, killing transcription even though we
always feed raw numpy arrays.

torchcodec is only a transitive dependency (pyannote-audio); no first-party
code decodes through it. When the real import fails, register a minimal stub
module so the transformers import succeeds and its `isinstance(...,
torchcodec.decoders.AudioDecoder)` check is False for our dict inputs. On
platforms where torchcodec imports cleanly this is a no-op.
"""

import logging
import sys
import types

logger = logging.getLogger(__name__)

_checked = False


class _StubModule(types.ModuleType):
    """Marker subclass so the stub is distinguishable from the real module."""


class _StubAudioDecoder:
    """Placeholder type; never instantiated. isinstance() against it is False."""


def ensure_torchcodec_importable_or_stub() -> bool:
    """Return True if real torchcodec is usable, False if the stub was installed."""
    global _checked
    if _checked:
        return not isinstance(sys.modules.get("torchcodec"), _StubModule)
    _checked = True
    try:
        import torchcodec  # noqa: F401

        return True
    except Exception as exc:  # RuntimeError from broken native libs, not ImportError
        logger.warning(
            "torchcodec is installed but failed to import (%s: %s); "
            "registering a stub so transformers array-input paths keep working.",
            type(exc).__name__,
            str(exc).splitlines()[0] if str(exc) else "",
        )
        stub = _StubModule("torchcodec")
        decoders = _StubModule("torchcodec.decoders")
        decoders.AudioDecoder = _StubAudioDecoder
        stub.decoders = decoders
        sys.modules["torchcodec"] = stub
        sys.modules["torchcodec.decoders"] = decoders
        return False
