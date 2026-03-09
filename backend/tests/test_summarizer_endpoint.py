import httpx

from app.config import Settings
from app.core.summarizer import _call_summarization_endpoint


def _response(status_code: int, url: str, **kwargs) -> httpx.Response:
    request = httpx.Request("POST", url)
    return httpx.Response(status_code, request=request, **kwargs)


def test_call_summarization_endpoint_retries_loading_responses(monkeypatch):
    settings = Settings(
        summarization_endpoint_url="http://summary-server:8080/v1",
        summarization_model="summary.gguf",
        summarization_endpoint_timeout_s=5.0,
    )
    calls = {"count": 0}
    sleeps = []

    def fake_post(url, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            return _response(
                503,
                url,
                json={"error": {"message": "Loading model", "type": "unavailable_error", "code": 503}},
            )
        return _response(200, url, json={"choices": [{"message": {"content": "{\"ok\": true}"}}]})

    monkeypatch.setattr("app.core.summarizer.httpx.post", fake_post)
    monkeypatch.setattr("app.core.summarizer.time.sleep", sleeps.append)

    result = _call_summarization_endpoint(
        settings,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
        temperature=0.1,
    )

    assert calls["count"] == 3
    assert sleeps == [0.5, 1.0]
    assert result["choices"][0]["message"]["content"] == "{\"ok\": true}"


def test_call_summarization_endpoint_retries_request_errors(monkeypatch):
    settings = Settings(
        summarization_endpoint_url="http://summary-server:8080/v1",
        summarization_model="summary.gguf",
        summarization_endpoint_timeout_s=5.0,
    )
    calls = {"count": 0}
    sleeps = []

    def fake_post(url, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("connection refused", request=httpx.Request("POST", url))
        return _response(200, url, json={"choices": []})

    monkeypatch.setattr("app.core.summarizer.httpx.post", fake_post)
    monkeypatch.setattr("app.core.summarizer.time.sleep", sleeps.append)

    result = _call_summarization_endpoint(
        settings,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
        temperature=0.1,
    )

    assert calls["count"] == 2
    assert sleeps == [0.5]
    assert result == {"choices": []}
