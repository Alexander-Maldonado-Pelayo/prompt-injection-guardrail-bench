"""
eval/harness.py — one evaluation loop that every detector plugs into.

A "predictor" is anything with a .predict(texts) -> list[int] method
(1 = injection, 0 = benign) and a .name attribute. See eval/predictors/.

What we measure, and why:
  accuracy / precision / recall / F1  — the usual
  fpr          — false-positive rate on ALL benign rows
  fpr_hard     — false-positive rate on benign rows that *look* like attacks
                 (every benign technique except the plain ones). This is the
                 number the dataset exists to measure.
  by_family    — recall per attack family, FPR per benign technique, so we
                 can see *which* kinds of inputs a detector gets wrong.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

# Benign techniques that are plain requests, not look-alikes.
# Everything else in the benign class counts as a hard negative.
PLAIN_BENIGN = {"benign", "plain", "plain_request", "none"}


class Predictor(Protocol):
    name: str

    def predict(self, texts: list[str]) -> list[int]: ...


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Metrics:
    predictor: str
    split: str
    n: int
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    fpr: float = 0.0
    fpr_hard: float = 0.0
    n_hard_negatives: int = 0
    recall_by_family: dict[str, float] = field(default_factory=dict)
    fpr_by_technique: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def evaluate(predictor: Predictor, rows: list[dict], split: str = "test") -> tuple[Metrics, list[int]]:
    """Run the predictor over the rows and score it. Returns (metrics, predictions)."""
    texts = [r["text"] for r in rows]
    t0 = time.time()
    preds = [int(p) for p in predictor.predict(texts)]
    elapsed = time.time() - t0
    assert len(preds) == len(rows), "predictor returned wrong number of labels"
    m = score(predictor.name, rows, preds, split)
    m.seconds = round(elapsed, 2)
    return m, preds


def score(name: str, rows: list[dict], preds: list[int], split: str = "test") -> Metrics:
    """Score a list of 0/1 predictions against the rows' labels."""
    m = Metrics(predictor=name, split=split, n=len(rows))

    fam_hit: dict[str, list[int]] = {}      # attack_family -> [1 if caught, 0 if missed]
    tech_fp: dict[str, list[int]] = {}      # benign technique -> [1 if falsely flagged]
    hard_fp = hard_n = 0

    for r, p in zip(rows, preds):
        y = int(r["label"])
        if y == 1:
            fam_hit.setdefault(r["attack_family"], []).append(int(p == 1))
            if p == 1:
                m.tp += 1
            else:
                m.fn += 1
        else:
            tech = r.get("technique", "benign")
            tech_fp.setdefault(tech, []).append(int(p == 1))
            if tech not in PLAIN_BENIGN:
                hard_n += 1
                hard_fp += int(p == 1)
            if p == 1:
                m.fp += 1
            else:
                m.tn += 1

    m.accuracy = _rate(m.tp + m.tn, m.n)
    m.precision = _rate(m.tp, m.tp + m.fp)
    m.recall = _rate(m.tp, m.tp + m.fn)
    m.f1 = _rate(2 * m.tp, 2 * m.tp + m.fp + m.fn)
    m.fpr = _rate(m.fp, m.fp + m.tn)
    m.fpr_hard = _rate(hard_fp, hard_n)
    m.n_hard_negatives = hard_n
    m.recall_by_family = {k: _rate(sum(v), len(v)) for k, v in sorted(fam_hit.items())}
    m.fpr_by_technique = {k: _rate(sum(v), len(v)) for k, v in sorted(tech_fp.items())}
    return m


def save(m: Metrics, preds: list[int] | None = None, rows: list[dict] | None = None) -> Path:
    """Write results/<predictor>__<split>.json (+ per-row predictions for error analysis)."""
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{m.predictor}__{m.split}.json"
    payload = asdict(m)
    if preds is not None and rows is not None:
        payload["rows"] = [
            {"id": r["id"], "label": r["label"], "pred": p,
             "attack_family": r["attack_family"], "technique": r.get("technique")}
            for r, p in zip(rows, preds)
        ]
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def summary_line(m: Metrics) -> str:
    return (f"{m.predictor:<22} acc={m.accuracy:.3f}  P={m.precision:.3f}  R={m.recall:.3f}  "
            f"F1={m.f1:.3f}  FPR={m.fpr:.3f}  FPR(hard)={m.fpr_hard:.3f}  "
            f"n={m.n}  {m.seconds}s")
