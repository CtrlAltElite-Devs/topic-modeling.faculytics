#!/usr/bin/env python3
"""
Main evaluation script for topic modeling pipeline.

Usage:
    python scripts/run_eval.py --dataset augmented --run-id 001
    python scripts/run_eval.py --dataset real --run-id 002 --force-embed
"""
import argparse
import json
import logging
import random
import numpy as np
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DATA_DIR, EXPERIMENTS_DIR,
    UC_DATASET_PATH, AUGMENTED_DATASET_PATH, UC_FILTERED_DATASET_PATH,
    LABSE_MODEL, DEVICE
)
from src.preprocess import clean_dataset
from src.embed import embed_texts
from src.topic_model import run_bertopic, extract_topic_info, get_topic_assignments
from src.evaluate import compute_metrics, format_metrics_report
from src.notify import post_to_discord

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# Default baseline parameters (Run 001)
DEFAULT_PARAMS = {
    "min_topic_size": 15,
    "nr_topics": None,  # Auto-detect
    "umap_n_neighbors": 15,
    "umap_n_components": 5
}


def load_dataset(dataset_type: str) -> list[str]:
    """Load dataset based on type."""
    if dataset_type == "augmented":
        logger.info(f"Loading augmented dataset from {AUGMENTED_DATASET_PATH}")
        with open(AUGMENTED_DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Extract text field
        return [item["text"] for item in data]

    elif dataset_type == "real":
        logger.info(f"Loading real dataset from {UC_DATASET_PATH}")
        import pandas as pd
        df = pd.read_csv(UC_DATASET_PATH)
        if "comment" in df.columns:
            return df["comment"].dropna().astype(str).tolist()
        else:
            text_cols = df.select_dtypes(include=["object"]).columns
            if len(text_cols) > 0:
                logger.warning(f"'comment' column not found, using '{text_cols[0]}'")
                return df[text_cols[0]].dropna().astype(str).tolist()
            raise ValueError("No text column found in dataset")

    elif dataset_type == "real_filtered":
        logger.info(f"Loading sentiment-gated real dataset from {UC_FILTERED_DATASET_PATH}")
        import json
        with open(UC_FILTERED_DATASET_PATH, encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Filter: negative+neutral always included, positive only if >=10 words")
        return [item["text"] for item in data]

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def save_run_artifacts(
    run_dir: Path,
    params: dict,
    metrics: dict,
    topic_info: list[dict],
    texts: list[str]
) -> None:
    """Save all run artifacts to the experiment directory."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save parameters
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    # Save metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save topics as markdown
    with open(run_dir / "topics.md", "w") as f:
        f.write(f"# Topics for Run {run_dir.name}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        for topic in topic_info:
            f.write(f"## Topic {topic['topic_id']}: {topic['label']}\n")
            f.write(f"- Documents: {topic['doc_count']}\n")
            f.write(f"- Keywords: {', '.join(topic['keywords'])}\n\n")

    # Save text count
    with open(run_dir / "info.json", "w") as f:
        json.dump({
            "n_texts": len(texts),
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"Artifacts saved to {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run topic modeling evaluation")
    parser.add_argument(
        "--dataset", 
        choices=["augmented", "real", "real_filtered"], 
        required=True,
        help="Dataset to use"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier (e.g., 001, 002)"
    )
    parser.add_argument(
        "--force-embed",
        action="store_true",
        help="Force regenerate embeddings even if cached"
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=DEFAULT_PARAMS["min_topic_size"],
        help="Minimum topic size"
    )
    parser.add_argument(
        "--nr-topics",
        type=int,
        default=None,
        help="Number of topics (None for auto)"
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=DEFAULT_PARAMS["umap_n_neighbors"],
        help="UMAP n_neighbors"
    )
    parser.add_argument(
        "--umap-n-components",
        type=int,
        default=DEFAULT_PARAMS["umap_n_components"],
        help="UMAP n_components"
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip Discord notification"
    )

    args = parser.parse_args()

    # Build params
    params = {
        "min_topic_size": args.min_topic_size,
        "nr_topics": args.nr_topics,
        "umap_n_neighbors": args.umap_n_neighbors,
        "umap_n_components": args.umap_n_components,
        "dataset": args.dataset,
        "run_id": args.run_id
    }

    # Global seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    logger.info(f"=== Starting Run {args.run_id} on {args.dataset} dataset ===")

    # Step 1: Load data
    texts_raw = load_dataset(args.dataset)
    logger.info(f"Loaded {len(texts_raw)} raw texts")

    # Step 2: Preprocess
    texts = clean_dataset(texts_raw)
    logger.info(f"After preprocessing: {len(texts)} texts")

    if len(texts) < 100:
        logger.error("Too few texts after preprocessing, aborting")
        return 1

    # Step 3: Embed
    embeddings = embed_texts(
        texts,
        batch_size=64,
        dataset_name=args.dataset,
        force=args.force_embed
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Step 4: Run BERTopic
    model = run_bertopic(embeddings, texts, params)
    topics = get_topic_assignments(model, texts, embeddings)
    topic_info = extract_topic_info(model)

    # Step 5: Evaluate (load LaBSE for embedding coherence metric)
    logger.info("Loading LaBSE for embedding coherence evaluation...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(LABSE_MODEL, device=DEVICE)
    metrics = compute_metrics(model, topics, texts, embeddings, embed_model=embed_model)

    # Step 6: Save artifacts
    run_dir = EXPERIMENTS_DIR / f"run_{args.run_id}"
    save_run_artifacts(run_dir, params, metrics, topic_info, texts)

    # Step 7: Format report
    report = format_metrics_report(metrics, topic_info, args.run_id, params)
    print("\n" + report + "\n")

    # Step 8: Post to Discord
    if not args.no_notify:
        logger.info("Posting results to Discord...")
        success = post_to_discord(report)
        if success:
            logger.info("Discord notification sent!")
        else:
            logger.warning("Discord notification failed")

    logger.info(f"=== Run {args.run_id} complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
