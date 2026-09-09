"""Ollama transport for the brain. Uses /api/chat with function-calling so qwen3
can drive the pipeline tools. Thinking models also return a `thinking` field."""

import json
import os

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("STUDIO_MODEL", "qwen3")


def available_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def chat(messages, tools=None, model=DEFAULT_MODEL, think=True, timeout=600):
    """One non-streaming chat turn. Returns the response `message` dict, which may
    contain `content`, `thinking`, and/or `tool_calls`."""
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0.6}}
    if tools:
        body["tools"] = tools
    if think:
        body["think"] = True
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json().get("message", {})
