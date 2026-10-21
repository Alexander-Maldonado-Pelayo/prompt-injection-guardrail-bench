"""
Llama Guard 3 (1B) via Ollama.  `ollama pull llama-guard3:1b`

Llama Guard is a *content-safety* classifier (violence, hate, etc.), not a
prompt-injection detector. It's in the benchmark on purpose: a common mistake
is assuming a safety guardrail covers OWASP LLM01. Expect low recall here.

It replies "safe" or "unsafe\nS<n>" for the user turn it's shown.
"""
from .ollama_base import OllamaPredictor, chat


class LlamaGuard3Predictor(OllamaPredictor):
    name = "llamaguard3-1b"
    model = "llama-guard3:1b"

    def classify_one(self, text: str) -> int:
        reply = chat(self.model, [{"role": "user", "content": text}]).strip().lower()
        return int(reply.startswith("unsafe"))
