import pytest

from app.config import Settings
from app.core import summarizer


def test_format_audio_context_for_summary_merges_structured_context_and_visual_corroboration():
    audio_context = {
        "primary_context": {
            "label": "broadcast or TV playback",
            "confidence": "high",
            "score": 0.84,
            "hint": (
                "Audio most likely sounds like television or broadcast playback rather than "
                "direct participant speech."
            ),
        },
        "supporting_contexts": [
            {
                "label": "noticeable music bed or soundtrack",
                "hint": "Audio includes noticeable music or soundtrack backing.",
            }
        ],
        "summary_context": (
            "Primary audio context: broadcast or TV playback (high confidence). "
            "Supporting audio cues: noticeable music bed or soundtrack."
        ),
    }
    frames = [
        {"timestamp": 12.0, "description": "TV news broadcast on a smart TV", "is_slide": False},
        {"timestamp": 24.0, "ocr_text": "Cable TV breaking news coverage", "is_slide": False},
        {"timestamp": 36.0, "title": "Television sports broadcast", "is_slide": False},
    ]

    result = summarizer.format_audio_context_for_summary(
        audio_context=audio_context,
        audio_hints=[
            "Audio most likely sounds like television or broadcast playback rather than direct participant speech.",
            "Room echo is mild.",
        ],
        frames=frames,
    )

    assert "Use these cues as supporting evidence only" in result
    assert "- Classifier summary: Primary audio context: broadcast or TV playback" in result
    assert "- Primary classification: broadcast or TV playback (high confidence, score 0.84)." in result
    assert "- Supporting classifier cues: noticeable music bed or soundtrack." in result
    assert "Additional audio hint: Room echo is mild." in result
    assert result.count("Additional audio hint:") == 1
    assert "Visual corroboration: Visuals show a TV/television playing content" in result


def test_extract_chat_content_strips_think_blocks():
    response = {
        "choices": [
            {
                "message": {
                    "content": "<think>\ninternal reasoning\n</think>\n\nFinal answer."
                }
            }
        ]
    }

    assert summarizer._extract_chat_content(response) == "Final answer."


@pytest.mark.asyncio
async def test_generate_summary_includes_audio_context_in_prompt(monkeypatch):
    captured = {}
    settings = Settings(
        summarization_endpoint_url="http://summary-server:8080/v1",
        summarization_model="summary.gguf",
        summary_chunk_minutes=6.0,
        summary_chunk_overlap_minutes=0.6,
    )

    def fake_call(settings, messages, max_tokens, temperature):
        captured["messages"] = messages
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"executive_summary":"ok","key_points":[],"action_items":[],'  # noqa: E501
                            '"decisions":[],"topics":[]}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(summarizer, "get_settings", lambda: settings)
    monkeypatch.setattr(summarizer, "_call_summarization_endpoint", fake_call)

    result = await summarizer.generate_summary(
        transcript={
            "text": "Reviewed the release checklist.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "speaker": "Speaker 1",
                    "text": "Reviewed the release checklist.",
                }
            ],
            "duration": 5.0,
        },
        audio_context={
            "primary_context": {
                "label": "direct software-demo narration",
                "confidence": "medium",
                "score": 0.44,
                "hint": (
                    "Audio most likely sounds like a presenter or participant directly "
                    "narrating a software demo or walkthrough."
                ),
            },
            "supporting_contexts": [
                {
                    "label": "noticeable music bed or soundtrack",
                    "hint": "Audio includes noticeable music or soundtrack backing.",
                }
            ],
            "summary_context": (
                "Primary audio context: direct software-demo narration (medium confidence). "
                "Supporting audio cues: noticeable music bed or soundtrack."
            ),
        },
        audio_hints=["Room echo is mild."],
        title="Release Demo",
    )

    prompt = captured["messages"][1]["content"]

    assert result["executive_summary"] == "ok"
    assert "## Audio Context" in prompt
    assert "Primary classification: direct software-demo narration" in prompt
    assert "Supporting classifier cues: noticeable music bed or soundtrack." in prompt
    assert "Additional audio hint: Room echo is mild." in prompt
