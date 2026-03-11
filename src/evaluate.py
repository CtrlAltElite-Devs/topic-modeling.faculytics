"""
Topic model evaluation metrics.
"""
import logging
from typing import Any
import numpy as np
from bertopic import BERTopic
from sklearn.metrics import silhouette_score
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

logger = logging.getLogger(__name__)


def compute_npmi_coherence(model: BERTopic, texts: list[str], top_n: int = 10) -> float:
    """
    Compute NPMI coherence score for topics.
    
    NPMI (Normalized Pointwise Mutual Information) measures how often
    topic words co-occur in the original documents.
    
    Target: > 0.1 indicates coherent topics.
    """
    # Tokenize texts
    tokenized = [text.lower().split() for text in texts]
    
    # Build dictionary
    dictionary = Dictionary(tokenized)
    
    # Get topic words
    topic_words = []
    for topic_id in model.get_topic_info()["Topic"]:
        if topic_id == -1:  # Skip outliers
            continue
        words = model.get_topic(topic_id)
        if words:
            topic_words.append([word for word, _ in words[:top_n]])
    
    if not topic_words:
        logger.warning("No topics found for coherence calculation")
        return 0.0
    
    # Compute NPMI coherence
    try:
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_npmi",
            window_size=10  # Standard sliding window for short texts (Lau et al. 2014)
        )
        return coherence_model.get_coherence()
    except Exception as e:
        logger.warning(f"Coherence calculation failed: {e}")
        return 0.0


def compute_topic_diversity(model: BERTopic, top_n: int = 10) -> float:
    """
    Compute topic diversity: ratio of unique words across all topics.
    
    Diversity = (unique words) / (total words across topics)
    Target: > 0.7 indicates diverse, non-redundant topics.
    """
    all_words = []
    
    for topic_id in model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        if words:
            all_words.extend([word for word, _ in words[:top_n]])
    
    if not all_words:
        return 0.0
    
    unique_words = set(all_words)
    return len(unique_words) / len(all_words)


def compute_outlier_ratio(model: BERTopic) -> float:
    """
    Compute the ratio of documents assigned to outlier topic (-1).
    
    Target: < 0.2 (less than 20% outliers).
    """
    topic_info = model.get_topic_info()
    outlier_row = topic_info[topic_info["Topic"] == -1]
    
    if outlier_row.empty:
        return 0.0
    
    outlier_count = outlier_row["Count"].values[0]
    total_count = topic_info["Count"].sum()
    
    return outlier_count / total_count


def compute_silhouette(topics: list[int], embeddings: np.ndarray) -> float:
    """
    Compute silhouette score for topic clusters.
    
    Measures how similar documents are to their own cluster vs other clusters.
    Range: -1 to 1, higher is better.
    """
    # Filter out outliers for silhouette calculation
    valid_mask = np.array(topics) != -1
    valid_topics = np.array(topics)[valid_mask]
    valid_embeddings = embeddings[valid_mask]
    
    # Need at least 2 clusters and 2 samples per cluster
    unique_topics = set(valid_topics)
    if len(unique_topics) < 2 or len(valid_topics) < 10:
        logger.warning("Not enough valid clusters for silhouette score")
        return 0.0
    
    try:
        return silhouette_score(valid_embeddings, valid_topics, metric="cosine")
    except Exception as e:
        logger.warning(f"Silhouette calculation failed: {e}")
        return 0.0


def compute_embedding_coherence(model: BERTopic, embed_model=None) -> float:
    """
    Compute embedding-based coherence: average pairwise cosine similarity
    between top keyword embeddings per topic.

    Language-agnostic alternative to NPMI — critical for multilingual models
    where co-occurrence metrics fail (Lau et al. 2014 coherence limitations).
    Target: > 0.5
    """
    if embed_model is None:
        return 0.0

    scores = []
    for topic_id in model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        if not words or len(words) < 2:
            continue
        keywords = [w for w, _ in words[:10]]
        try:
            vecs = embed_model.encode(keywords, show_progress_bar=False, normalize_embeddings=True)
            # Average pairwise cosine similarity (dot product since normalized)
            n = len(vecs)
            sim_sum = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    sim_sum += float(np.dot(vecs[i], vecs[j]))
                    count += 1
            if count > 0:
                scores.append(sim_sum / count)
        except Exception:
            continue

    return round(float(np.mean(scores)), 4) if scores else 0.0


def compute_metrics(
    model: BERTopic,
    topics: list[int],
    texts: list[str],
    embeddings: np.ndarray,
    embed_model=None
) -> dict[str, Any]:
    """
    Compute all evaluation metrics for a topic model.
    
    Returns:
        Dict with: npmi_coherence, topic_diversity, outlier_ratio, 
                   num_topics, silhouette_score
    """
    logger.info("Computing evaluation metrics...")
    
    # Number of topics (excluding outlier topic)
    num_topics = len(set(topics)) - (1 if -1 in topics else 0)
    
    metrics = {
        "embedding_coherence": compute_embedding_coherence(model, embed_model=embed_model),
        "npmi_coherence": round(compute_npmi_coherence(model, texts), 4),
        "topic_diversity": round(compute_topic_diversity(model), 4),
        "outlier_ratio": round(compute_outlier_ratio(model), 4),
        "num_topics": num_topics,
        "silhouette_score": round(compute_silhouette(topics, embeddings), 4),
        "total_documents": len(texts),
        "outlier_count": sum(1 for t in topics if t == -1)
    }
    
    logger.info(f"Metrics: NPMI={metrics['npmi_coherence']}, "
                f"Diversity={metrics['topic_diversity']}, "
                f"Outliers={metrics['outlier_ratio']:.1%}")
    
    return metrics


def format_metrics_report(
    metrics: dict[str, Any],
    topic_info: list[dict],
    run_id: str,
    params: dict[str, Any]
) -> str:
    """
    Format metrics and topics into a Discord-friendly report.
    Uses bullet lists instead of tables.
    """
    lines = [
        f"## 📊 Topic Modeling Run: {run_id}",
        "",
        "**Parameters:**",
        f"• min_topic_size: {params.get('min_topic_size', 'N/A')}",
        f"• nr_topics: {params.get('nr_topics', 'auto')}",
        f"• umap_n_neighbors: {params.get('umap_n_neighbors', 'N/A')}",
        f"• umap_n_components: {params.get('umap_n_components', 'N/A')}",
        "",
        "**Metrics:**",
        f"• NPMI Coherence: {metrics['npmi_coherence']} (target: > 0.1)",
        f"• Topic Diversity: {metrics['topic_diversity']} (target: > 0.7)",
        f"• Outlier Ratio: {metrics['outlier_ratio']:.1%} (target: < 20%)",
        f"• Silhouette Score: {metrics['silhouette_score']}",
        f"• Number of Topics: {metrics['num_topics']}",
        f"• Total Documents: {metrics['total_documents']}",
        f"• Outlier Count: {metrics['outlier_count']}",
        "",
        "**Quality Check:**",
    ]
    
    # Quality indicators
    npmi_ok = metrics['npmi_coherence'] > 0.1
    div_ok = metrics['topic_diversity'] > 0.7
    outlier_ok = metrics['outlier_ratio'] < 0.2
    
    lines.append(f"• Coherence: {'✅ PASS' if npmi_ok else '❌ FAIL'}")
    lines.append(f"• Diversity: {'✅ PASS' if div_ok else '❌ FAIL'}")
    lines.append(f"• Outliers: {'✅ PASS' if outlier_ok else '❌ FAIL'}")
    
    overall = "✅ ALL TARGETS MET" if (npmi_ok and div_ok and outlier_ok) else "⚠️ NEEDS TUNING"
    lines.append(f"• Overall: {overall}")
    
    # Top topics
    lines.extend(["", "**Top Topics:**"])
    for topic in topic_info[:10]:  # Show top 10
        keywords = ", ".join(topic["keywords"][:5])
        lines.append(f"• Topic {topic['topic_id']} ({topic['doc_count']} docs): {keywords}")
    
    if len(topic_info) > 10:
        lines.append(f"• ... and {len(topic_info) - 10} more topics")
    
    return "\n".join(lines)
