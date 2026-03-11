"""
Text preprocessing for multilingual student feedback.
Handles Cebuano, Tagalog, English, and code-switching.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns for cleaning
EXCEL_ARTIFACT_PATTERN = re.compile(r'^#(NAME|VALUE|REF|DIV/0|NULL|NUM|N/A)\??$', re.IGNORECASE)
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
LAUGHTER_PATTERN = re.compile(r'\b(ha){2,}\b|\b(he){2,}\b|\b(hi){2,}\b|\blol+\b|\blmao+\b|\brofl+\b', re.IGNORECASE)
REPEATED_CHAR_PATTERN = re.compile(r'(.)\1{2,}')
PUNCTUATION_SPAM_PATTERN = re.compile(r'([!?.]){3,}')
BROKEN_EMOJI_PATTERN = re.compile(r'\ufffd+')
KEYBOARD_MASH_PATTERN = re.compile(r'^[asdfghjklqwertyuiopzxcvbnm]{5,}$', re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r'\s+')


def clean_text(text: str) -> Optional[str]:
    """
    Clean a single feedback text entry.
    
    Returns:
        Cleaned text, or None if the entry should be dropped.
    """
    if not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # Drop empty
    if not text:
        return None
    
    # Drop Excel artifacts
    if EXCEL_ARTIFACT_PATTERN.match(text):
        return None
    
    # Strip URLs
    text = URL_PATTERN.sub('', text)
    
    # Strip broken emoji (replacement character)
    text = BROKEN_EMOJI_PATTERN.sub('', text)
    
    # Strip laughter noise
    text = LAUGHTER_PATTERN.sub('', text)
    
    # Reduce repeated characters (3+ → 1)
    # e.g., "soooo gooood" → "so god"
    text = REPEATED_CHAR_PATTERN.sub(r'\1', text)
    
    # Reduce punctuation spam (3+ → single)
    text = PUNCTUATION_SPAM_PATTERN.sub(r'\1', text)
    
    # Normalize whitespace
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    
    # Drop if empty after cleaning
    if not text:
        return None
    
    # Drop pure gibberish (keyboard mash detection)
    # Only letters, no spaces, looks like random key presses
    text_no_space = text.replace(' ', '').lower()
    if len(text_no_space) >= 5 and KEYBOARD_MASH_PATTERN.match(text_no_space):
        # Additional check: if it has no vowels or very low vowel ratio, likely gibberish
        vowels = sum(1 for c in text_no_space if c in 'aeiou')
        if vowels / len(text_no_space) < 0.15:
            return None
    
    # Drop entries with fewer than 3 words after cleaning
    words = text.split()
    if len(words) < 3:
        return None
    
    return text


def clean_dataset(texts: list[str]) -> list[str]:
    """
    Clean a list of feedback texts, filtering out None entries.
    
    Args:
        texts: List of raw feedback strings.
        
    Returns:
        List of cleaned strings (dropped entries removed).
    """
    original_count = len(texts)
    cleaned = []
    
    for text in texts:
        result = clean_text(text)
        if result is not None:
            cleaned.append(result)
    
    drop_count = original_count - len(cleaned)
    drop_pct = (drop_count / original_count * 100) if original_count > 0 else 0
    
    logger.info(f"Preprocessing: {original_count} → {len(cleaned)} entries "
                f"(dropped {drop_count}, {drop_pct:.1f}%)")
    
    return cleaned


def clean_dataset_with_indices(texts: list[str]) -> tuple[list[str], list[int]]:
    """
    Clean dataset and return both cleaned texts and their original indices.
    Useful for maintaining alignment with other data (e.g., labels).
    
    Returns:
        Tuple of (cleaned_texts, original_indices)
    """
    cleaned = []
    indices = []
    
    for i, text in enumerate(texts):
        result = clean_text(text)
        if result is not None:
            cleaned.append(result)
            indices.append(i)
    
    logger.info(f"Preprocessing: {len(texts)} → {len(cleaned)} entries "
                f"(dropped {len(texts) - len(cleaned)})")
    
    return cleaned, indices
