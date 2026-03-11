"""
BERTopic wrapper for topic modeling with pre-computed embeddings.
"""
import logging
from typing import Any
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)


def run_bertopic(
    embeddings: np.ndarray,
    texts: list[str],
    params: dict[str, Any]
) -> BERTopic:
    """
    Run BERTopic with pre-computed embeddings.
    
    Args:
        embeddings: Pre-computed LaBSE embeddings (n_samples, 768).
        texts: Original text documents.
        params: Dict with keys:
            - min_topic_size: Minimum cluster size (default 15)
            - nr_topics: Number of topics or None for auto
            - umap_n_neighbors: UMAP n_neighbors (default 15)
            - umap_n_components: UMAP output dimensions (default 5)
            
    Returns:
        Fitted BERTopic model.
    """
    # Extract params with defaults
    min_topic_size = params.get("min_topic_size", 15)
    nr_topics = params.get("nr_topics", None)  # None = auto
    umap_n_neighbors = params.get("umap_n_neighbors", 15)
    umap_n_components = params.get("umap_n_components", 5)
    
    logger.info(f"BERTopic params: min_topic_size={min_topic_size}, nr_topics={nr_topics}, "
                f"umap_n_neighbors={umap_n_neighbors}, umap_n_components={umap_n_components}")
    
    # Configure UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )
    
    # Configure HDBSCAN for clustering
    # min_samples controls outlier sensitivity — lower = fewer outliers
    # McInnes et al. (2017): min_samples defaults to min_cluster_size, but
    # decoupling allows tighter clusters with fewer rejected docs
    min_samples = params.get("hdbscan_min_samples", 5)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )
    
    # Vectorizer for c-TF-IDF (multilingual-friendly)
    # Stop words: English + high-frequency Cebuano/Tagalog function words
    # These are grammatical tokens that dominate c-TF-IDF without adding topic signal
    MULTILINGUAL_STOP_WORDS = list({
        # Role/title words — appear in almost all feedback, no topic signal
        # Adding these breaks up the generic "propesor/estudyante" catch-all cluster
        "propesor", "estudyante", "guro", "magaaral", "maestra", "maestro",
        "students", "student", "maam", "sir", "professor", "teacher", "instructor",
        "faculty", "teacher", "atty", "miss",
        # English stop words (core subset)
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "that", "this",
        "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
        "your", "he", "she", "they", "his", "her", "their", "as", "not",
        "also", "so", "if", "than", "then", "when", "what", "how", "all",
        "very", "just", "more", "about", "up", "out", "no", "her", "him",
        # Cebuano function words + possessives + filler
        "ang", "nga", "sa", "ni", "si", "ug", "og", "kay", "ba", "na",
        "man", "lang", "ra", "jud", "gyud", "ko", "mo", "mi", "siya",
        "niya", "nako", "imo", "iya", "ato", "nato", "namo", "kamo",
        "sila", "nila", "kang", "kanang", "kanila", "dili", "wala",
        "naa", "adto", "diri", "didto", "pero",
        # Cebuano possessives (iyang/kanyang = his/her, among/atong = our)
        "iyang", "kanyang", "among", "atong", "ilang", "inyong", "among",
        # High-frequency Cebuano filler (kaayo=very, mao=that/it, bawat=every)
        "kaayo", "mao", "bawat", "aralin", "klase", "gawain",
        # Generic action verbs (high-freq across all topics, zero discrimination)
        # nagbibigay=gives, naguse=uses, nagtatakda=sets/assigns
        "nagbibigay", "naguse", "nagtatakda", "naggagamit", "ginagamit",
        "nagbibigay ng", "ginagamit ng",
        # Generic English filler found in Topic 13 (sentiment noise)
        "really", "much", "am", "way", "feel", "truly",
        # Generic time/context words
        "every", "during", "specific", "even",
        # Tagalog function words
        "ng", "mga", "ay", "nang", "rin", "din", "po", "ho", "yung",
        "kasi", "naman", "lang", "pa", "pag", "kung", "dahil", "para",
        "hindi", "at", "o", "ni", "niya", "siya", "namin", "natin",
        "nila", "ito", "iyon", "yon", "dito", "doon",
    })
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=MULTILINGUAL_STOP_WORDS,
        min_df=2
    )
    
    # Representation model: KeyBERTInspired (Grootendorst 2022, §4.2)
    # Uses embedding similarity instead of c-TF-IDF frequency counts.
    # Language-agnostic — finds keywords semantically closest to cluster centroid.
    # Eliminates the need for manual stop word lists as primary mechanism.
    use_keybert = params.get("use_keybert", True)
    representation_model = KeyBERTInspired() if use_keybert else None

    # Create BERTopic model
    # embedding_model=None tells BERTopic to use pre-computed embeddings
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=nr_topics,
        verbose=True
    )
    
    # Fit with pre-computed embeddings
    logger.info(f"Fitting BERTopic on {len(texts)} documents...")
    topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
    
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)  # Exclude outlier topic
    n_outliers = sum(1 for t in topics if t == -1)
    outlier_pct = n_outliers / len(topics) * 100
    
    logger.info(f"Found {n_topics} topics, {n_outliers} outliers ({outlier_pct:.1f}%)")
    
    return topic_model


def extract_topic_info(model: BERTopic) -> list[dict]:
    """
    Extract topic information from a fitted BERTopic model.
    
    Returns:
        List of dicts with: topic_id, label, keywords, doc_count
    """
    topic_info = model.get_topic_info()
    
    results = []
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        
        # Skip outlier topic (-1) in the detailed list
        if topic_id == -1:
            continue
        
        # Get top keywords
        topic_words = model.get_topic(topic_id)
        keywords = [word for word, _ in topic_words[:10]] if topic_words else []
        
        results.append({
            "topic_id": topic_id,
            "label": row.get("Name", f"Topic_{topic_id}"),
            "keywords": keywords,
            "doc_count": row["Count"]
        })
    
    return results


def get_topic_assignments(model: BERTopic, texts: list[str], embeddings: np.ndarray) -> list[int]:
    """
    Get topic assignments for documents.
    
    Returns:
        List of topic IDs (-1 for outliers).
    """
    topics, _ = model.transform(texts, embeddings=embeddings)
    return topics
