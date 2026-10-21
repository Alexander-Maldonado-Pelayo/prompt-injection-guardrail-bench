"""
eval/report.py — turn results/*.json into a markdown table and the FPR chart
for the README and the LinkedIn post.

    python -m eval.report            # uses results/*__test.json
    python -m eval.report --split validation
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    files = sorted(RESULTS.glob(f"*__{args.split}.json"))
    if not files:
        raise SystemExit(f"no results for split '{args.split}' in {RESULTS}")
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    rows.sort(key=lambda r: (-r["f1"], r["fpr_hard"]))

    # ---- markdown table
    md = ["| Detector | Acc | Precision | Recall | F1 | FPR (all benign) | FPR (hard negatives) | n |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['predictor']} | {r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} "
                  f"| {r['f1']:.3f} | {r['fpr']:.1%} | {r['fpr_hard']:.1%} | {r['n']} |")
    table = "\n".join(md)
    (RESULTS / f"table__{args.split}.md").write_text(table, encoding="utf-8")
    print(table)

    # ---- chart: recall vs hard-negative FPR, one bar pair per detector
    names = [r["predictor"] for r in rows]
    recall = [r["recall"] * 100 for r in rows]
    fpr_hard = [r["fpr_hard"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    x = range(len(names))
    w = 0.38
    ax.bar([i - w / 2 for i in x], recall, w, label="Recall on attacks (%)", color="#3b6ea5")
    ax.bar([i + w / 2 for i in x], fpr_hard, w, label="False-positive rate on hard negatives (%)", color="#c0504d")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_title(f"Prompt-injection detectors on guardrail-hard-negatives ({args.split} split)")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    for i, (r_, f_) in enumerate(zip(recall, fpr_hard)):
        ax.text(i - w / 2, r_ + 1, f"{r_:.0f}", ha="center", fontsize=8)
        ax.text(i + w / 2, f_ + 1, f"{f_:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = RESULTS / f"recall_vs_fpr__{args.split}.png"
    fig.savefig(out)
    print("->", out.relative_to(REPO))


if __name__ == "__main__":
    main()
