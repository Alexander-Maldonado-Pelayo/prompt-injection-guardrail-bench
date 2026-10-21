"""
Registry of detectors. Each module defines a class with:
    name: str
    predict(texts: list[str]) -> list[int]     # 1 = injection, 0 = benign

Add a new detector by writing a module in this folder and registering it below.
Imports are lazy so you don't need every dependency installed to run one detector.
"""
from importlib import import_module

REGISTRY = {
    # name            : (module, class)
    "heuristic":        ("eval.predictors.heuristic", "HeuristicPredictor"),
    "protectai":        ("eval.predictors.protectai", "ProtectAIPredictor"),
    "promptguard2":     ("eval.predictors.promptguard", "PromptGuard2Predictor"),
    "llamaguard3":      ("eval.predictors.ollama_guard", "LlamaGuard3Predictor"),
    "qwen3-zeroshot":   ("eval.predictors.ollama_zeroshot", "QwenZeroShotPredictor"),
    "finetuned":        ("eval.predictors.finetuned", "FinetunedPredictor"),
}


def get(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown predictor '{name}'. Known: {', '.join(REGISTRY)}")
    module, cls = REGISTRY[name]
    return getattr(import_module(module), cls)(**kwargs)
