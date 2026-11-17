"""Проверки через настоящий Instructor с подменённым HTTP, без вызовов LLM."""

import json

import httpx
import instructor
from instructor.core import InstructorRetryException
from openai import APIStatusError, OpenAI
import pytest

from src.nlp.clinical_extractor import extract_relations
from src.nlp.clinical_relations import ClinicalFragment


@pytest.fixture
def fragment():
    return ClinicalFragment(
        document_id="synthetic-example",
        location="Страница 2, учебный раздел",
        page_number=2,
        text="Препарат А не противопоказан при заболевании Б\nпри условии В.",
    )


def make_payload(fragment):
    return {
        "relations": [{
            "subject": {"name": "Препарат А", "kind": "drug"},
            "object": {"name": "заболевании Б", "kind": "disease"},
            "relation_type": "CONTRAINDICATED_FOR",
            "polarity": "negated",
            "conditions": ["при условии В"],
            "description": "Учебный пример отрицания противопоказания",
            "source": {
                "document_id": fragment.document_id,
                "location": fragment.location,
                "page_number": fragment.page_number,
                "quote": fragment.text,
            },
            "status": "pending_review",
        }],
    }


def run_extraction(fragment, payload, requests, *, status_code=200):
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(status_code, json={
            "id": "test-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            }],
        })

    with OpenAI(
        api_key="test-key",
        base_url="https://example.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as openai_client:
        client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON)
        return extract_relations(fragment, client=client, model="test-model")


def test_extracts_with_instructor_and_preserves_evidence(fragment):
    requests = []
    result = run_extraction(fragment, make_payload(fragment), requests)
    relation = result.relations[0]

    assert len(requests) == 1
    assert requests[0]["temperature"] == 0
    sent_fragment = json.loads(requests[0]["messages"][-1]["content"])
    assert sent_fragment["text"] == fragment.text
    assert relation.source.quote == fragment.text
    assert relation.polarity == "negated"
    assert relation.conditions == ["при условии В"]
    assert relation.status == "unresolved"


def test_uses_source_metadata_from_input(fragment):
    payload = make_payload(fragment)
    payload["relations"][0]["source"].update(
        document_id="invented-document", location="invented-section", page_number=99,
    )

    result = run_extraction(fragment, payload, [])

    source = result.relations[0].source
    assert source.document_id == fragment.document_id
    assert source.location == fragment.location
    assert source.page_number == fragment.page_number


@pytest.mark.parametrize("change", ["missing_quote", "uncertain"])
def test_marks_disputed_results(fragment, change):
    payload = make_payload(fragment)
    relation = payload["relations"][0]
    if change == "missing_quote":
        relation["source"]["quote"] = "Этой цитаты нет в исходном тексте."
    else:
        relation["polarity"] = "uncertain"

    result = run_extraction(fragment, payload, [])

    assert result.relations[0].status == "disputed"
    assert result.relations[0].source.quote == relation["source"]["quote"]


def test_restores_source_line_breaks_in_model_quote(fragment):
    payload = make_payload(fragment)
    payload["relations"][0]["source"]["quote"] = fragment.text.replace("\n", " ")

    result = run_extraction(fragment, payload, [])

    assert result.relations[0].source.quote == fragment.text
    assert result.relations[0].status == "unresolved"


def test_returns_empty_result_when_no_relations(fragment):
    result = run_extraction(fragment, {"relations": []}, [])

    assert result.relations == []


def test_invalid_llm_response_is_not_an_empty_result(fragment):
    requests = []
    with pytest.raises(InstructorRetryException):
        run_extraction(fragment, {"relations": [{"status": "confirmed"}]}, requests)

    assert len(requests) > 1


def test_api_error_is_not_an_empty_result(fragment):
    with pytest.raises((APIStatusError, InstructorRetryException)):
        run_extraction(fragment, {}, [], status_code=503)
