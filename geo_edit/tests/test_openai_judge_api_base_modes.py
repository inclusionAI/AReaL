from types import SimpleNamespace

from geo_edit.evaluation import openai_as_judge as judge_module
from geo_edit.evaluation.openai_as_judge import OpenAIJudge


def _patch_openai_client(monkeypatch, *, responses_text: str = "Score: 1", chat_text: str = "Score: 0"):
    calls = {"responses": 0, "chat": 0, "client_kwargs": None}

    class _Responses:
        def create(self, **kwargs):
            calls["responses"] += 1
            return SimpleNamespace(output_text=responses_text)

    class _ChatCompletions:
        def create(self, **kwargs):
            calls["chat"] += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=chat_text))])

    class _Client:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs
            self.responses = _Responses()
            self.chat = SimpleNamespace(completions=_ChatCompletions())

    monkeypatch.setattr(judge_module, "OpenAI", _Client)
    return calls


def test_auto_mode_uses_responses_for_llm_proxy(monkeypatch):
    calls = _patch_openai_client(monkeypatch)

    judge = OpenAIJudge(
        api_key="test-key",
        model="gpt-5-mini",
        api_base="https://llm-proxy.perflab.nvidia.com/openai/v1",
    )
    score = judge.judge_correctness("q", "gt", "pred")

    assert score == "1"
    assert calls["responses"] == 1
    assert calls["chat"] == 0
    assert calls["client_kwargs"] == {
        "api_key": "test-key",
        "base_url": "https://llm-proxy.perflab.nvidia.com/openai/v1",
    }


def test_auto_mode_uses_chat_for_matrix(monkeypatch):
    calls = _patch_openai_client(monkeypatch)

    judge = OpenAIJudge(
        api_key="test-key",
        model="gpt-4o",
        api_base="https://matrixllm.alipay.com/v1",
    )
    score = judge.judge_correctness("q", "gt", "pred")

    assert score == "0"
    assert calls["responses"] == 0
    assert calls["chat"] == 1
    assert calls["client_kwargs"] == {
        "api_key": "test-key",
        "base_url": "https://matrixllm.alipay.com/v1",
    }


def test_base_url_is_optional(monkeypatch):
    calls = _patch_openai_client(monkeypatch)

    judge = OpenAIJudge(
        api_key="test-key",
        model="gpt-5-mini",
    )
    score = judge.judge_correctness("q", "gt", "pred")

    assert score == "1"
    assert calls["responses"] == 1
    assert calls["chat"] == 0
    assert calls["client_kwargs"] == {"api_key": "test-key"}
