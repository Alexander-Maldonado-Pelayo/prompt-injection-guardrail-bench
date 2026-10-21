"""
eval/run.py — run one or more detectors through the harness.

    python -m eval.run --predictor heuristic
    python -m eval.run --predictor protectai promptguard2 --split test
    python -m eval.run --predictor finetuned --model-id xRainbowdartx/deberta-injection-hardneg
    python -m eval.run --predictor heuristic --data data/sample_preview.jsonl   # offline smoke test

Results land in results/<name>__<split>.json
"""
import argparse
from pathlib import Path

from eval import harness
from eval.predictors import get, REGISTRY

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", nargs="+", required=True, choices=list(REGISTRY))
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--data", help="explicit JSONL path (overrides --split)")
    ap.add_argument("--model-id", help="override model id (HF repo, local path, or Ollama tag)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    path = Path(args.data) if args.data else REPO / "data" / f"{args.split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found — run `python -m data.load` first")
    rows = harness.load_jsonl(path)
    split = path.stem

    for name in args.predictor:
        kwargs = {}
        if args.model_id:
            kwargs["model_id" if name not in ("llamaguard3", "qwen3-zeroshot") else "model"] = args.model_id
        if name in ("protectai", "promptguard2", "finetuned"):
            kwargs["threshold"] = args.threshold
        if name in ("llamaguard3", "qwen3-zeroshot"):
            kwargs["verbose"] = args.verbose

        print(f"\n== {name} on {split} ({len(rows)} rows) ==")
        pred = get(name, **kwargs)
        m, preds = harness.evaluate(pred, rows, split=split)
        out = harness.save(m, preds, rows)
        print(harness.summary_line(m))
        print("recall by family:", m.recall_by_family)
        print("FPR by benign technique:", m.fpr_by_technique)
        print("->", out.relative_to(REPO))


if __name__ == "__main__":
    main()
