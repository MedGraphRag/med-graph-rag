"""Настройки Ollama и формат запросов через настоящий SDK с подменённым HTTP."""

import json
from unittest.mock import MagicMock, Mock

import httpx
from openai import OpenAI
import pytest

from src.nlp import clinical_llm
from src.nlp.clinical_extractor import extract_relations
from src.nlp.clinical_relations import ClinicalFragment


@pytest.fixture
def isolated_env(monkeypatch):
    monkeypatch.setattr(clinical_llm, "load_dotenv", lambda *args: None)
    for name in ("LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_TIMEOUT", "LLM_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("reasoning_effort", [None, "none", "low", " "])
def test_ollama_default_request_without_api_key(isolated_env, monkeypatch, reasoning_effort):
    if reasoning_effort is not None:
        monkeypatch.setenv("LLM_REASONING_EFFORT", reasoning_effort)
    requests = []
    options = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "ollama-test", "object": "chat.completion", "created": 0,
            "model": "gemma4:12b",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": '{"relations": []}',
            }}],
        })

    def make_client(**kwargs):
        options.append(kwargs)
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(clinical_llm, "OpenAI", make_client)
    fragment = ClinicalFragment(document_id="test", location="test", page_number=1, text="Пример.")
    with clinical_llm.extraction_client() as (client, model):
        result = extract_relations(fragment, client=client, model=model)

    assert result.relations == []
    assert str(requests[0].url) == "http://localhost:11434/v1/chat/completions"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "gemma4:12b"
    assert payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}
    if reasoning_effort and reasoning_effort.strip():
        assert payload["reasoning_effort"] == reasoning_effort
    else:
        assert "reasoning_effort" not in payload
    assert options[0]["api_key"] == "ollama"
    assert options[0]["timeout"] == 300


def test_respects_custom_endpoint_model_and_timeout(isolated_env, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://server:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("LLM_TIMEOUT", "600")
    monkeypatch.setenv("LLM_API_KEY", "custom-key")
    constructor = MagicMock()
    monkeypatch.setattr(clinical_llm, "OpenAI", constructor)
    monkeypatch.setattr(clinical_llm.instructor, "from_openai", Mock())

    with clinical_llm.extraction_client() as (_, model):
        assert model == "custom-model"

    assert constructor.call_args.kwargs == {
        "base_url": "http://server:11434/v1", "api_key": "custom-key",
        "timeout": 600, "max_retries": 0,
    }
    constructor.return_value.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize("name,value", [
    ("LLM_MODEL", " "), ("LLM_BASE_URL", ""), ("LLM_TIMEOUT", "0"),
])
def test_rejects_invalid_settings(isolated_env, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        with clinical_llm.extraction_client():
            pytest.fail("Invalid settings must not create a client")
