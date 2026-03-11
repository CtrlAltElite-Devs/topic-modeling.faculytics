# TUNING_LOG.md — Y.A.L.A.'s Topic Modeling Memory

> This file tracks experiment runs, parameter tuning decisions, and lessons learned.
> Updated by Y.A.L.A. after each evaluation run.

---

## Status

**Current State:** Environment setup — no eval runs yet

**Last Updated:** 2026-03-12

---

## Baseline Parameters (Run 001)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `min_topic_size` | 15 | BERTopic default; balances granularity vs. noise. Smaller values may capture niche topics but increase outliers. HDBSCAN recommendation: 10-25 for medium datasets. |
| `nr_topics` | None (auto) | Let HDBSCAN determine natural cluster count first. Manual reduction via BERTopic's hierarchical merging can follow. |
| `umap_n_neighbors` | 15 | LaBSE paper uses 15 as default for semantic similarity tasks. Larger values (30+) smooth local structure but may merge distinct topics. |
| `umap_n_components` | 5 | BERTopic paper (Grootendorst, 2022) recommends 5 for topic modeling. Lower dims (2-3) lose information; higher (10+) adds noise for HDBSCAN. |

### Paper Citations

- **LaBSE:** Feng et al., "Language-agnostic BERT Sentence Embedding" (ACL 2022) — 768-dim embeddings, cosine similarity optimized
- **BERTopic:** Grootendorst, "BERTopic: Neural topic modeling with a class-based TF-IDF procedure" (arXiv 2022)
- **Coherence Metrics:** Röder et al., "Exploring the Space of Topic Coherence Measures" (WSDM 2015) — NPMI preferred for interpretability
- **Topic Diversity:** Dieng et al., "Topic Modeling in Embedding Spaces" (TACL 2020)

---

## Metric Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| NPMI Coherence | > 0.1 | Industry standard for "coherent" topics. 0.0-0.1 is marginal. |
| Topic Diversity | > 0.7 | Ensures topics are distinct, not redundant keyword overlaps. |
| Outlier Ratio | < 0.2 | More than 20% outliers suggests poor clustering or preprocessing issues. |
| Silhouette Score | > 0.0 | Positive indicates meaningful clusters; aim for > 0.1 |

---

## Dataset Notes

### Augmented Dataset (`feedback_augmented_v1.json`)
- **Size:** ~7,000 entries
- **Structure:** `{text, label, lang_type}`
- **Languages:** Cebuano, Tagalog, English, mixed code-switching
- **Quality:** Curated, labeled — good for baseline testing
- **Use case:** Run 001 baseline, hyperparameter tuning

### Real Dataset (`uc_dataset_20krows1.csv`)
- **Size:** ~34,000 rows (claimed 20k in filename, actually 34k)
- **Structure:** CSV with `comment` column
- **Quality:** Raw student feedback — expect noise, Excel artifacts, gibberish
- **Use case:** Final production evaluation after tuning

---

## Known Preprocessing Issues

1. **Excel artifacts:** `#NAME?`, `#VALUE!` — must drop
2. **Broken emoji:** Unicode replacement chars (`\ufffd`) — strip
3. **Laughter spam:** `hahaha`, `hehe`, `lol` variants — strip (noise, not signal)
4. **Keyboard mash:** Random letter strings — drop entries
5. **Code-switching:** "Nindot kaayo ang teacher pero minsan boring" — preserve! This is valid multilingual feedback
6. **Short entries:** Single words like "ok", "nice" — drop (< 3 words)

---

## Experiment Log

### Run 001 — Baseline (PENDING)
- **Dataset:** augmented
- **Parameters:** min_topic_size=15, nr_topics=auto, umap_n_neighbors=15, umap_n_components=5
- **Status:** Not yet executed
- **Command:** `python scripts/run_eval.py --dataset augmented --run-id 001`

---

## Tuning Strategy

1. **Run 001:** Baseline on augmented dataset with default params
2. **Analyze:** Check outlier ratio first — if > 30%, reduce min_topic_size to 10
3. **Run 002:** Adjust based on Run 001 metrics
4. **Run 003+:** Grid search on promising params
5. **Final:** Best params on real dataset

---

## Next Steps

- [ ] Activate venv and install dependencies
- [ ] Run 001 baseline on augmented dataset
- [ ] Post results to Discord
- [ ] Analyze metrics and plan Run 002
