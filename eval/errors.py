"""
eval/errors.py — dump the misclassified rows for one or more result files
so you can read them and write the findings section.

    python -m eval.errors results/protectai-v2__test.json
    python -m eval.errors results/*__test.json --only fp     # false positives only

Writes results/errors__<name>__<split>.md next to each result file.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from eval.harness import load_jsonl

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--only", choices=["fp", "fn"], help="restrict to one error type")
    args = ap.parse_args()

    for rp in args.results:
        res = json.loads(Path(rp).read_text(encoding="utf-8"))
        if "rows" not in res:
            print(f"{rp}: no per-row predictions saved, skipping")
            continue
        data_path = REPO / "data" / f"{res['split']}.jsonl"
        text_by_id = {r["id"]: r["text"] for r in load_jsonl(data_path)}

        fps = [r for r in res["rows"] if r["label"] == 0 and r["pred"] == 1]
        fns = [r for r in res["rows"] if r["label"] == 1 and r["pred"] == 0]

        lines = [f"# Errors — {res['predictor']} on {res['split']}", "",
                 f"False positives: {len(fps)}   False negatives: {len(fns)}", ""]

        if args.only != "fn":
            lines += ["## False positives (benign flagged as injection)", "",
                      "By technique: " + ", ".join(f"{k}={v}" for k, v in Counter(r["technique"] for r in fps).most_common()),
                      ""]
            for r in fps:
                lines.append(f"- **[{r['technique']}]** id={r['id']} — {text_by_id.get(r['id'], '?')}")
            lines.append("")

        if args.only != "fp":
            lines += ["## False negatives (injection missed)", "",
                      "By family: " + ", ".join(f"{k}={v}" for k, v in Counter(r["attack_family"] for r in fns).most_common()),
                      ""]
            for r in fns:
                lines.append(f"- **[{r['attack_family']}/{r['technique']}]** id={r['id']} — {text_by_id.get(r['id'], '?')}")

        out = Path(rp).with_name(f"errors__{res['predictor']}__{res['split']}.md")
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"{res['predictor']}: {len(fps)} FP, {len(fns)} FN -> {out}")


if __name__ == "__main__":
    main()
