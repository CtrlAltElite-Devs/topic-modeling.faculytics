"""
LaBSE embedding generation with caching.
"""
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import LABSE_MODEL, DEVICE, EMBEDDINGS_CACHE_DIR

logger = logging.getLogger(__name__)


def get_cache_path(dataset_name: str) -> Path:
    """Get the cache file path for a dataset."""
    if dataset_name:
        return EMBEDDINGS_CACHE_DIR / f"embeddings_cache_{dataset_name}.npy"
    return EMBEDDINGS_CACHE_DIR / "embeddings_cache.npy"


def embed_texts(
    texts: list[str],
    batch_size: int = 64,
    dataset_name: str = "",
    force: bool = False
) -> np.ndarray:
    """
    Generate LaBSE embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for encoding (default 64 for 6GB VRAM).
        dataset_name: Name for cache file (e.g., "augmented", "real").
        force: If True, regenerate embeddings even if cache exists.
        
    Returns:
        np.ndarray of shape (len(texts), 768) — LaBSE embedding dimension.
    """
    cache_path = get_cache_path(dataset_name)
    
    # Check cache
    if not force and cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)
        
        # Validate cache matches current data
        if embeddings.shape[0] == len(texts):
            logger.info(f"Loaded {embeddings.shape[0]} embeddings from cache")
            return embeddings
        else:
            logger.warning(f"Cache size mismatch ({embeddings.shape[0]} vs {len(texts)}), regenerating...")
    
    # Load model
    logger.info(f"Loading LaBSE model on {DEVICE}...")
    model = SentenceTransformer(LABSE_MODEL, device=DEVICE)
    
    # Generate embeddings with progress bar
    logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize for cosine similarity
    )
    
    # Save to cache
    logger.info(f"Saving embeddings to {cache_path}")
    np.save(cache_path, embeddings)
    
    return embeddings


def load_cached_embeddings(dataset_name: str) -> np.ndarray | None:
    """
    Load embeddings from cache if available.
    
    Returns:
        Embeddings array or None if not cached.
    """
    cache_path = get_cache_path(dataset_name)
    if cache_path.exists():
        return np.load(cache_path)
    return None
