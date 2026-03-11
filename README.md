# faculytics-topic-modeling

Multilingual topic modeling pipeline for student feedback analysis. Handles Cebuano, Tagalog, English, and code-switching using LaBSE embeddings and BERTopic.

## What It Does

1. **Preprocesses** raw student feedback (cleans noise, drops gibberish)
2. **Embeds** text using LaBSE (multilingual sentence embeddings)
3. **Discovers topics** using BERTopic (UMAP + HDBSCAN + c-TF-IDF)
4. **Evaluates** with NPMI coherence, topic diversity, silhouette score
5. **Reports** results to Discord for async review

## Tech Stack

- Python 3.11
- LaBSE via sentence-transformers (768-dim multilingual embeddings)
- BERTopic for topic modeling
- CUDA GPU support (RTX 2060, 6GB VRAM)
- Discord notifications via bot token

## Setup

### 1. Clone and enter repo

```bash
cd /home/yander/Documents/codes/faculytics/topic-modeling.faculytics
```

### 2. Create virtual environment with uv

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install sentence-transformers bertopic umap-learn hdbscan gensim scikit-learn
uv pip install pandas numpy tqdm requests python-dotenv
```

### 4. Configure Discord

Create `.env` in repo root:

```env
DISCORD_BOT_TOKEN=your_bot_token_here
```

### 5. Add datasets

Place in `data/`:
- `uc_dataset_20krows1.csv` — real UC student feedback
- `feedback_augmented_v1.json` — curated augmented dataset

## Usage

### Run evaluation

```bash
# Baseline run on augmented dataset
python scripts/run_eval.py --dataset augmented --run-id 001

# Run on real dataset with custom params
python scripts/run_eval.py --dataset real --run-id 002 \
    --min-topic-size 10 \
    --umap-n-neighbors 20

# Force re-embed (ignore cache)
python scripts/run_eval.py --dataset augmented --run-id 003 --force-embed

# Skip Discord notification
python scripts/run_eval.py --dataset augmented --run-id test --no-notify
```

### Sleep-trigger workflow

For async runs (Y.A.L.A. triggers, Yander sleeps):

1. Y.A.L.A. runs: `python scripts/run_eval.py --dataset augmented --run-id 001`
2. Results auto-post to Discord channel `#topic-modeling-results`
3. Yander reviews in the morning, provides feedback
4. Y.A.L.A. adjusts params, runs next iteration

## Repo Structure

```
topic-modeling.faculytics/
├── src/
│   ├── config.py        # Constants, paths, device selection
│   ├── preprocess.py    # Text cleaning for multilingual feedback
│   ├── embed.py         # LaBSE embedding with caching
│   ├── topic_model.py   # BERTopic wrapper
│   ├── evaluate.py      # Metrics: NPMI, diversity, silhouette
│   └── notify.py        # Discord posting
├── scripts/
│   └── run_eval.py      # Main evaluation entry point
├── data/
│   ├── uc_dataset_20krows1.csv
│   ├── feedback_augmented_v1.json
│   └── embeddings_cache_*.npy  # Auto-generated
├── experiments/
│   └── run_001/         # Per-run artifacts
│       ├── params.json
│       ├── metrics.json
│       └── topics.md
├── .env                 # Discord bot token (gitignored)
├── TUNING_LOG.md        # Y.A.L.A.'s persistent memory
└── README.md
```

## Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| NPMI Coherence | > 0.1 | Topic word co-occurrence quality |
| Topic Diversity | > 0.7 | Uniqueness of keywords across topics |
| Outlier Ratio | < 20% | Documents not assigned to any topic |
| Silhouette Score | > 0.0 | Cluster separation quality |

## Notes

- LaBSE handles code-switching naturally (trained on 109 languages)
- First embed run downloads ~2GB model, cached thereafter
- Embeddings cached per dataset — delete `data/embeddings_cache_*.npy` to regenerate
- Experiment artifacts saved to `experiments/run_{id}/`
