import json
import sys
import requests
from .config import OLLAMA_URL, MODEL

SYSTEM_PROMPT = (
    "You are a helpful assistant designed to output JSON. "
    "Always respond with valid JSON only."
)


def chat(prompt: str, timeout: int = 300) -> str | None:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "think": False,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def chat_stream(prompt: str, model: str | None = None) -> str | None:
    """Streaming chat — prints tokens in real time. No timeout."""
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model or MODEL,
            "think": False,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        stream=True,
    )
    resp.raise_for_status()
    full = ""
    for line in resp.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            full += token
            sys.stdout.write(token)
            sys.stdout.flush()
    print()  # newline at the end
    return full


def chat_json(prompt: str, timeout: int = 300) -> dict | list | str | None:
    raw = chat(prompt, timeout=timeout)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw  # return raw string for substring fallback


def chat_json_stream(prompt: str, model: str | None = None) -> dict | list | str | None:
    """Streaming version of chat_json — prints tokens live."""
    raw = chat_stream(prompt, model=model)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
