"""Meta Llama Prompt Guard 2 (86M). Gated repo — accept the license on the
model page and run `huggingface-cli login` first. Binary: LABEL_0 benign, LABEL_1 malicious."""
from .hf_classifier import HFClassifierPredictor


class PromptGuard2Predictor(HFClassifierPredictor):
    name = "promptguard2-86m"
    model_id = "meta-llama/Llama-Prompt-Guard-2-86M"
    positive_labels = {"LABEL_1", "MALICIOUS", "JAILBREAK", "INJECTION"}
