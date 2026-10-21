"""
Keyword/regex baseline. Deliberately naive — this is the "what a WAF rule
would do" floor that every real detector should beat, and it runs with no
dependencies so the harness can be tested offline.
"""
import re

PATTERNS = [
    r"\bignore (all |any |the )?(previous|prior|above|earlier) (instructions|messages|context)\b",
    r"\b(system prompt|system message|initial instructions|original instructions)\b",
    r"\b(repeat|print|show|reveal|paste|output) (the |your )?(text above|instructions|prompt)\b",
    r"\b(do not|don't|without) (ask(ing)?|surface|confirm)",
    r"\b(pre-?approved|self-?approve|delegated (approval|authority)|authorized to)\b",
    r"\b(transfer_funds|update_config|install_package|exfiltrate|change_billing_email)\b",
    r"\[INTERNAL SYSTEM NOTE",
    r"<!--.*?-->",
    r"\b(persist|store|remember) (this |it )?(to|in) (memory|scratchpad)\b",
    r"\bact as (an? )?unrestricted\b",
]
RX = [re.compile(p, re.IGNORECASE) for p in PATTERNS]


class HeuristicPredictor:
    name = "heuristic"

    def predict(self, texts: list[str]) -> list[int]:
        return [int(any(r.search(t) for r in RX)) for t in texts]
