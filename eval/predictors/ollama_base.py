"""
Shared Ollama client. Talks to the local Ollama server over HTTP, no SDK needed.
Start Ollama on Windows and `ollama pull <model>` before running.
"""
from __future__ import annotations

import json
import os
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def chat(model: str, messages: list[dict], **options) -> str:
    """Single non-streaming chat call. Returns the assistant text."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, **options},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)["message"]["content"]


class OllamaPredictor:
    """Subclasses implement classify_one(text) -> int."""
    name = "ollama"
    model = ""

    def __init__(self, model: str | None = None, verbose: bool = False):
        self.model = model or self.model
        self.verbose = verbose

    def classify_one(self, text: str) -> int:
        raise NotImplementedError

    def predict(self, texts: list[str]) -> list[int]:
        out = []
        for i, t in enumerate(texts, 1):
            out.append(self.classify_one(t))
            if self.verbose and i % 25 == 0:
                print(f"  {self.name}: {i}/{len(texts)}")
        return out
