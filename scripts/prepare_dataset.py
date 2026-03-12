"""
prepare_dataset.py — Regenerate derived datasets from raw source files.

Usage:
    python scripts/prepare_dataset.py                    # regenerates all
    python scripts/prepare_dataset.py --dataset filtered # only real_filtered
    python scripts/prepare_dataset.py --min-words 10     # adjust word threshold

Datasets:
    filtered — sentiment-gated real dataset (production baseline)
               negative + neutral always included
               positive only if >= MIN_WORDS words
               source: data/uc_dataset_20krows1.csv
               output: data/uc_dataset_filtered.json
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
DATA = BASE / "data"

# Label encoding in uc_dataset_20krows1.csv
LABEL_POSITIVE = 1.0
LABEL_NEGATIVE = -1.0
LABEL_NEUTRAL = 0.0


def build_filtered(min_words: int = 10):
    """
    Sentiment-gated real dataset.
    Filters out short positive feedback (generic praise noise).
    """
    source = DATA / "uc_dataset_20krows1.csv"
    output = DATA / "uc_dataset_filtered.json"

    if not source.exists():
        log.error(f"Source file not found: {source}")
        log.error("Ensure uc_dataset_20krows1.csv is present in data/ (symlinked or copied).")
        return False

    try:
        import pandas as pd
    except ImportError:
        log.error("pandas not installed. Run: pip install pandas")
        return False

    df = pd.read_csv(source)
    log.info(f"Loaded {len(df)} rows from {source.name}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["wc"] = df["comment"].fillna("").str.split().str.len()
    df = df.dropna(subset=["comment"])

    total = len(df)
    pos_short = df[(df["label"] == LABEL_POSITIVE) & (df["wc"] < min_words)]
    pos_long  = df[(df["label"] == LABEL_POSITIVE) & (df["wc"] >= min_words)]
    neg       = df[df["label"] == LABEL_NEGATIVE]
    neu       = df[df["label"] == LABEL_NEUTRAL]

    log.info(f"Label breakdown (total {total}):")
    log.info(f"  Positive short (<{min_words} words) — EXCLUDED: {len(pos_short)} ({len(pos_short)/total*100:.1f}%)")
    log.info(f"  Positive long  (>={min_words} words) — included: {len(pos_long)} ({len(pos_long)/total*100:.1f}%)")
    log.info(f"  Negative                             — included: {len(neg)} ({len(neg)/total*100:.1f}%)")
    log.info(f"  Neutral                              — included: {len(neu)} ({len(neu)/total*100:.1f}%)")

    keep = pd.concat([pos_long, neg, neu])
    records = [
        {"text": row["comment"], "label": str(row["label"]), "lang_type": "mixed"}
        for _, row in keep.iterrows()
    ]

    output.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    log.info(f"✅ Saved {len(records)} entries → {output}")
    log.info(f"   Filter: positive <{min_words} words removed ({len(pos_short)/total*100:.1f}% of data)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Regenerate derived datasets")
    parser.add_argument(
        "--dataset",
        choices=["filtered", "all"],
        default="all",
        help="Which dataset to regenerate (default: all)"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=10,
        help="Minimum word count for positive feedback to be included (default: 10)"
    )
    args = parser.parse_args()

    if args.dataset in ("filtered", "all"):
        ok = build_filtered(min_words=args.min_words)
        if not ok:
            raise SystemExit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()
