"""Our own fine-tuned classifier (see train/finetune.py).
Pass model_id= to point at a local folder or your HF repo, e.g.
    python -m eval.run --predictor finetuned --model-id xRainbowdartx/deberta-injection-hardneg"""
from .hf_classifier import HFClassifierPredictor


class FinetunedPredictor(HFClassifierPredictor):
    name = "finetuned"
    model_id = "./train/out/best"          # default: local training output
    positive_labels = {"injection", "LABEL_1"}
