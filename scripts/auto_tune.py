#!/usr/bin/env python3
"""
Autonomous hyperparameter tuner for the topic modeling pipeline.

Runs a predefined grid of experiments, evaluates each, selects the best,
and posts updates to Discord after every run. No human intervention needed.

Usage:
    python scripts/auto_tune.py --dataset augmented --start-run 004
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, EXPERIMENTS_DIR, DISCORD_CHANNEL_ID
from src.preprocess import clean_dataset
from src.embed import embed_texts
from src.topic_model import run_bertopic, extract_topic_info, get_topic_assignments
from src.evaluate import (
    compute_metrics, format_metrics_report,
    compute_embedding_coherence, compute_topic_diversity, compute_outlier_ratio
)
from src.notify import post_to_discord

import json as _json
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ─── Parameter grid ─────────────────────────────────────────────────────────
# Each entry is a named experiment with rationale.
# Reasoning: Grootendorst (2022) — BERTopic paper
#   - nr_topics reduces via hierarchical merging (principled, not arbitrary)
#   - min_topic_size ~0.3-0.5% of corpus is the recommended range
#   - HDBSCAN min_samples < min_cluster_size reduces outlier rate
#
# Target: num_topics 10-25 (interpretable for educators), outlier < 20%,
#         embedding coherence > 0.5, diversity > 0.7

PARAM_GRID = [
    {
        "run_suffix": "004",
        "rationale": "Force merge to 20 topics via nr_topics. Tests hierarchical reduction (Grootendorst 2022 §3.4).",
        "params": {
            "min_topic_size": 20,
            "nr_topics": 20,
            "umap_n_neighbors": 15,
            "umap_n_components": 5,
            "hdbscan_min_samples": 5,
        }
    },
    {
        "run_suffix": "005",
        "rationale": "Target 15 topics (educator-interpretable count). Larger min_topic_size to ensure quality.",
        "params": {
            "min_topic_size": 25,
            "nr_topics": 15,
            "umap_n_neighbors": 15,
            "umap_n_components": 5,
            "hdbscan_min_samples": 8,
        }
    },
    {
        "run_suffix": "006",
        "rationale": "Wider UMAP neighborhood (n=20) for better global structure. Auto topics. Tests if topology helps.",
        "params": {
            "min_topic_size": 25,
            "nr_topics": None,
            "umap_n_neighbors": 20,
            "umap_n_components": 8,
            "hdbscan_min_samples": 8,
        }
    },
    {
        "run_suffix": "007",
        "rationale": "Aggressive topic reduction to 10 — minimum interpretable set for a faculty dashboard.",
        "params": {
            "min_topic_size": 30,
            "nr_topics": 10,
            "umap_n_neighbors": 15,
            "umap_n_components": 5,
            "hdbscan_min_samples": 10,
        }
    },
]


def compute_score(metrics: dict) -> float:
    """
    Composite score for ranking runs.
    Primary: embedding coherence (language-agnostic)
    Secondary: diversity, low outliers, reasonable topic count
    """
    coherence = metrics.get("embedding_coherence", 0)
    diversity = metrics.get("topic_diversity", 0)
    outlier_penalty = max(0, metrics.get("outlier_ratio", 1) - 0.2)  # Penalty above 20%
    topic_count = metrics.get("num_topics", 0)

    # Prefer 10–25 topics (educator-friendly range)
    if 10 <= topic_count <= 25:
        topic_bonus = 0.1
    else:
        topic_bonus = 0

    score = (coherence * 0.5) + (diversity * 0.3) - (outlier_penalty * 0.5) + topic_bonus
    return round(score, 4)


def load_dataset(dataset_type: str) -> list[str]:
    if dataset_type == "augmented":
        path = DATA_DIR / "feedback_augmented_v1.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [item["text"] for item in data]
    else:
        import pandas as pd
        df = pd.read_csv(DATA_DIR / "uc_dataset_20krows1.csv")
        return df["comment"].dropna().astype(str).tolist()


def save_artifacts(run_dir: Path, params: dict, metrics: dict, topic_info: list, texts: list):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(run_dir / "topics.md", "w") as f:
        f.write(f"# Topics — Run {run_dir.name}\nGenerated: {datetime.now().isoformat()}\n\n")
        for topic in topic_info:
            f.write(f"## Topic {topic['topic_id']}: {topic['label']}\n")
            f.write(f"- Documents: {topic['doc_count']}\n")
            f.write(f"- Keywords: {', '.join(topic['keywords'])}\n\n")


def format_run_report(experiment: dict, metrics: dict, topic_info: list, score: float, run_id: str) -> str:
    p = experiment["params"]
    lines = [
        f"🔬 **Auto-tune Run {run_id}**",
        f"📝 {experiment['rationale']}",
        "",
        "**Params:**",
        f"• min_topic_size: {p['min_topic_size']}",
        f"• nr_topics: {p.get('nr_topics', 'auto')}",
        f"• umap_n_neighbors: {p['umap_n_neighbors']}",
        f"• hdbscan_min_samples: {p.get('hdbscan_min_samples', 5)}",
        "",
        "**Metrics:**",
        f"• Embedding Coherence: {metrics.get('embedding_coherence', 0):.4f} (target: > 0.5)",
        f"• Topic Diversity: {metrics.get('topic_diversity', 0):.4f} (target: > 0.7)",
        f"• Outlier Ratio: {metrics.get('outlier_ratio', 0):.1%} (target: < 20%)",
        f"• Num Topics: {metrics.get('num_topics', 0)}",
        f"• Silhouette: {metrics.get('silhouette_score', 0):.4f}",
        "",
        "**Quality:**",
        f"• Coherence: {'✅' if metrics.get('embedding_coherence', 0) > 0.5 else '❌'}",
        f"• Diversity: {'✅' if metrics.get('topic_diversity', 0) > 0.7 else '❌'}",
        f"• Outliers: {'✅' if metrics.get('outlier_ratio', 1) < 0.2 else '❌'}",
        f"• Topics in range (10-25): {'✅' if 10 <= metrics.get('num_topics', 0) <= 25 else '❌'}",
        f"• **Composite Score: {score}**",
        "",
        "**Top Topics:**",
    ]
    for t in topic_info[:8]:
        kws = ", ".join(t["keywords"][:5])
        lines.append(f"• [{t['doc_count']} docs] {kws}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["augmented", "real"], default="augmented")
    parser.add_argument("--start-run", type=int, default=4)
    args = parser.parse_args()

    post_to_discord(f"🚀 **Auto-tune starting** — {len(PARAM_GRID)} experiments queued on `{args.dataset}` dataset. I'll post results after each run.")

    # Load + preprocess once
    logger.info("Loading dataset...")
    texts_raw = load_dataset(args.dataset)
    texts = clean_dataset(texts_raw)
    logger.info(f"Cleaned texts: {len(texts)}")

    # Embed once (cached)
    embeddings = embed_texts(texts, batch_size=64, dataset_name=args.dataset)
    logger.info(f"Embeddings: {embeddings.shape}")

    # Load sentence model for embedding coherence
    from sentence_transformers import SentenceTransformer
    from src.config import LABSE_MODEL, DEVICE
    embed_model = SentenceTransformer(LABSE_MODEL, device=DEVICE)

    best_score = -999
    best_run = None
    results_summary = []

    run_num = args.start_run
    for experiment in PARAM_GRID:
        run_id = f"{run_num:03d}"
        logger.info(f"\n{'='*50}\nRun {run_id}: {experiment['rationale']}\n{'='*50}")

        params = {**experiment["params"], "dataset": args.dataset, "run_id": run_id}

        try:
            # BERTopic
            model = run_bertopic(embeddings, texts, params)
            topics = get_topic_assignments(model, texts, embeddings)
            topic_info = extract_topic_info(model)

            # Metrics (using embedding coherence as primary)
            metrics = compute_metrics(model, topics, texts, embeddings, embed_model=embed_model)

            # Composite score
            score = compute_score(metrics)
            metrics["composite_score"] = score

            # Save
            run_dir = EXPERIMENTS_DIR / f"run_{run_id}"
            save_artifacts(run_dir, params, metrics, topic_info, texts)

            # Report to Discord
            report = format_run_report(experiment, metrics, topic_info, score, run_id)
            post_to_discord(report)

            results_summary.append({
                "run_id": run_id,
                "score": score,
                "num_topics": metrics["num_topics"],
                "outlier_ratio": metrics["outlier_ratio"],
                "embedding_coherence": metrics.get("embedding_coherence", 0),
            })

            if score > best_score:
                best_score = score
                best_run = run_id

        except Exception as e:
            logger.error(f"Run {run_id} failed: {e}", exc_info=True)
            post_to_discord(f"❌ Run {run_id} failed: {str(e)[:200]}")

        run_num += 1
        time.sleep(2)  # Brief pause between runs

    # Final summary
    summary_lines = ["🏆 **Auto-tune Complete**", ""]
    summary_lines.append("**All runs:**")
    for r in results_summary:
        flag = "👑" if r["run_id"] == best_run else "  "
        summary_lines.append(
            f"{flag} Run {r['run_id']}: score={r['score']:.3f} | "
            f"topics={r['num_topics']} | outliers={r['outlier_ratio']:.1%} | "
            f"coherence={r['embedding_coherence']:.3f}"
        )
    summary_lines.append("")
    summary_lines.append(f"**Best run: {best_run}** (score: {best_score:.3f})")
    summary_lines.append("<@735312299602214923> Review results above and confirm best params.")

    post_to_discord("\n".join(summary_lines))
    logger.info(f"Auto-tune complete. Best run: {best_run}")


if __name__ == "__main__":
    main()
