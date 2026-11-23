"""Клиент извлечения через Ollama или совместимый API, без Neo4j."""

from contextlib import contextmanager
import os
from pathlib import Path

from dotenv import load_dotenv
import instructor
from openai import OpenAI


@contextmanager
def extraction_client():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    model = os.getenv("LLM_MODEL", "gemma4:12b").strip()
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").strip()
    timeout = float(os.getenv("LLM_TIMEOUT", "300"))
    if not model or not base_url or timeout <= 0:
        raise ValueError("Заполните LLM_MODEL, LLM_BASE_URL; LLM_TIMEOUT должен быть > 0")
    client_options = {}
    reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "").strip()
    if reasoning_effort:
        client_options["reasoning_effort"] = reasoning_effort
    if model == "inclusionai/ling-3.2-flash:free":
        client_options["extra_body"] = {"reasoning": {"enabled": False}}

    with OpenAI(
        api_key=os.getenv("LLM_API_KEY") or "ollama",
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    ) as openai_client:
        client = instructor.from_openai(
            openai_client, mode=instructor.Mode.JSON, **client_options,
        )
        yield client, model
