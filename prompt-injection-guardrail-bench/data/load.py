"""
data/load.py — pull the dataset from Hugging Face, keep English rows,
write local JSONL splits and a profile report.

Run once from the repo root:
    python -m data.load            # English only (default)
    python -m data.load --all-languages

Outputs:
    data/train.jsonl, data/validation.jsonl, data/test.jsonl
    data/profile.md
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

DATASET = "3nesdeniz/guardrail-hard-negatives"
SPLITS = ["train", "validation", "test"]
HERE = Path(__file__).parent


def fetch_split(split: str) -> pd.DataFrame:
    """Download one parquet split straight from the Hub.

    pandas can read hf:// URIs once huggingface_hub is installed.
    """
    uri = f"hf://datasets/{DATASET}/data/{split}.parquet"
    return pd.read_parquet(uri)


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def profile(frames: dict[str, pd.DataFrame]) -> str:
    """Build a markdown profile of the splits we're going to train/evaluate on."""
    lines = ["# Dataset profile", "", f"Source: `{DATASET}`", ""]

    for split, df in frames.items():
        lines += [f"## {split} — {len(df)} rows", ""]

        # class balance
        c = Counter(df["class"])
        lines += ["| class | rows | share |", "|---|---:|---:|"]
        for k, v in c.most_common():
            lines.append(f"| {k} | {v} | {v / len(df):.1%} |")
        lines.append("")

        # attack families (injection rows only)
        inj = df[df["label"] == 1]
        lines += ["**Attack families (injection rows)**", "",
                  "| attack_family | rows |", "|---|---:|"]
        for k, v in Counter(inj["attack_family"]).most_common():
            lines.append(f"| {k} | {v} |")
        lines.append("")

        # benign techniques — this is the "hard negative" taxonomy
        ben = df[df["label"] == 0]
        lines += ["**Benign techniques (hard-negative types)**", "",
                  "| technique | rows |", "|---|---:|"]
        for k, v in Counter(ben["technique"]).most_common():
            lines.append(f"| {k} | {v} |")
        lines.append("")

        # severity + length
        lines += ["**Severity (injection rows)**: "
                  + ", ".join(f"{k}={v}" for k, v in Counter(inj["severity"]).most_common()),
                  "",
                  f"**Text length (chars)**: min={df['text'].str.len().min()}, "
                  f"median={int(df['text'].str.len().median())}, "
                  f"max={df['text'].str.len().max()}",
                  ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-languages", action="store_true",
                    help="keep Turkish rows too (default: English only)")
    args = ap.parse_args()

    frames = {}
    for split in SPLITS:
        df = fetch_split(split)
        if not args.all_languages:
            df = df[df["language"] == "en"].reset_index(drop=True)
        frames[split] = df
        out = HERE / f"{split}.jsonl"
        write_jsonl(df, out)
        print(f"{split}: {len(df)} rows -> {out}")

    (HERE / "profile.md").write_text(profile(frames), encoding="utf-8")
    print("wrote data/profile.md")


if __name__ == "__main__":
    main()
