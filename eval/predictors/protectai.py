"""ProtectAI prompt-injection detector (DeBERTa-v3-base), off the shelf.
Labels: SAFE / INJECTION."""
from .hf_classifier import HFClassifierPredictor


class ProtectAIPredictor(HFClassifierPredictor):
    name = "protectai-v2"
    model_id = "protectai/deberta-v3-base-prompt-injection-v2"
    positive_labels = {"INJECTION"}
