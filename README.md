# NLP Assignment 3 — Transformers + RAG
**CS-4063 Natural Language Processing | FAST NUCES**  
**Student:** Fatima Siddiqa &nbsp;|&nbsp; **ID:** 23i-2543

---

## Overview

A three-stage NLP pipeline built entirely from scratch (no pretrained models):

| Stage | Component | Description |
|-------|-----------|-------------|
| A | Encoder-Only Transformer | Multi-task: sentiment classification + helpfulness prediction |
| B | Retrieval Module | Cosine similarity search over CLS embeddings |
| C | Decoder-Only Transformer | Autoregressive explanation generation (RAG-grounded) |

---

## Results Summary

| Metric | Value |
|--------|-------|
| Sentiment Accuracy (test) | See `results/partA_learning_curves.png` |
| Helpfulness Accuracy (test) | See `results/partA_confusion_matrices.png` |
| Retrieval mean top-1 cosine sim | 0.984 |
| Sentiment agreement @ k=3 | ~76% |
| Decoder Perplexity — Full RAG | **7.88** |
| Decoder Perplexity — No RAG | 8.00 |
| RAG improvement | +0.12 perplexity points |

---

## Repository Structure

```
NLP-Assignment3/
├── i232543-NLP-Assignment3.ipynb   ← complete notebook (all parts)
├── models/
│   ├── encoder_best.pt             ← trained encoder weights
│   └── decoder_best.pt             ← trained decoder weights
├── results/
│   ├── vocabulary.pkl              ← built from training data only
│   ├── train_data.pkl              ← preprocessed train split
│   ├── val_data.pkl                ← preprocessed val split
│   ├── test_data.pkl               ← preprocessed test split
│   ├── train_embeddings.npy        ← CLS embeddings (31499 × 128)
│   ├── train_texts.pkl
│   ├── train_sent_labels.npy
│   ├── train_help_labels.npy
│   ├── test_retrieval_context.pkl  ← pre-computed top-3 context per test review
│   ├── train_history.json          ← encoder training curves
│   ├── dec_train_history.json      ← decoder training curves
│   ├── partA_learning_curves.png
│   ├── partA_confusion_matrices.png
│   ├── partB_retrieval_analysis.png
│   ├── partC_learning_curves.png
│   └── partC_rag_ablation.png
├── data/                           ← NOT tracked (too large — see below)
└── .gitignore
```

> **Note:** Raw JSON data files are excluded from the repo (each is 137–411 MB). Model weights and all preprocessed artefacts are committed so **you do not need to retrain or re-preprocess anything.**

---

## Quick Start — Inference Without Retraining

All trained weights and preprocessed artefacts are already in the repo. Follow these steps to run inference directly.

### 1. Clone the repo and mount Drive

```python
# Run Cell 1 in the notebook — it handles clone/pull automatically
# Requires a GitHub token stored in Colab Secrets as GITHUB_TOKEN2
```

### 2. Copy model weights from Google Drive

The model weights are backed up on the owner's Google Drive at `/content/drive/MyDrive/NLP_A3/`. If you have access, run:

```python
import shutil
shutil.copy('/content/drive/MyDrive/NLP_A3/encoder_best.pt',
            '/content/NLP-Assignment3/models/encoder_best.pt')
shutil.copy('/content/drive/MyDrive/NLP_A3/decoder_best.pt',
            '/content/NLP-Assignment3/models/decoder_best.pt')
```

Alternatively, if the weights are already committed to the repo, they will be present at `models/` after cloning.

### 3. Load preprocessed data (no raw JSON needed)

```python
import pickle

with open(f'{REPO_PATH}/results/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open(f'{REPO_PATH}/results/val_data.pkl', 'rb') as f:
    val_data = pickle.load(f)
with open(f'{REPO_PATH}/results/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

vocab = Vocabulary.load(f'{REPO_PATH}/results/vocabulary.pkl')
print(f'Loaded {len(train_data):,} train | {len(val_data):,} val | {len(test_data):,} test')
```

### 4. Load the encoder

```python
model = EncoderTransformer(
    vocab_size  = len(vocab),   # 35,749
    d_model     = 128,
    n_heads     = 4,
    n_layers    = 3,
    d_ff        = 256,
    max_seq_len = 128,
    dropout     = 0.1,
    pad_idx     = 0
).to(DEVICE)

model.load_state_dict(torch.load(
    f'{REPO_PATH}/models/encoder_best.pt', map_location=DEVICE
))
model.eval()
print('Encoder loaded.')
```

### 5. Load the decoder

```python
dec_model = DecoderTransformer(
    vocab_size  = len(vocab),   # 35,749
    d_model     = 128,
    n_heads     = 4,
    n_layers    = 3,
    d_ff        = 256,
    max_seq_len = 296,          # DEC_MAX_INPUT_LEN(256) + DEC_MAX_TARGET_LEN(40)
    dropout     = 0.1,
    pad_idx     = 0
).to(DEVICE)

dec_model.load_state_dict(torch.load(
    f'{REPO_PATH}/models/decoder_best.pt', map_location=DEVICE
))
dec_model.eval()
print('Decoder loaded.')
```

### 6. Run the full pipeline on a new review

```python
review = "This product is amazing! Great quality and fast shipping."

# Step 1 — Encode + predict
q_emb, results = retrieve_for_review(review, k=3)

# Step 2 — Generate explanation
explanation = generate_explanation(
    review_text = review,
    sentiment   = 2,        # 0=Negative, 1=Neutral, 2=Positive
    helpfulness = 1,        # 0=Not helpful, 1=Helpful
    retrieved   = results,
)
print('Explanation:', explanation)
```

---

## Dataset

Raw data is **not included** in this repo due to file size (750 MB total). Download from the official Amazon Reviews dataset:

**Source:** https://nijianmo.github.io/amazon/index.html

Files used:
- `Beauty_5.json` (~137 MB)
- `Home_and_Kitchen_5.json` (~411 MB)
- `Sports_and_Outdoors_5.json` (~203 MB)

Place them in `data/` before running the data loading cells. However, since `results/train_data.pkl`, `results/val_data.pkl`, and `results/test_data.pkl` are already committed, **raw data is only needed if you want to re-run preprocessing from scratch.**

---

## Model Architecture

### Encoder (Part A)
| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 3 |
| d_ff | 256 |
| dropout | 0.1 |
| max_seq_len | 128 |
| vocab_size | 35,749 |
| Tasks | Sentiment (3-class) + Helpfulness (binary) |
| Loss weights | λ_sentiment=0.7, λ_helpfulness=0.3 |
| Optimizer | AdamW, lr=3e-4, cosine schedule |
| Epochs | 10 |

### Retrieval (Part B)
| Parameter | Value |
|-----------|-------|
| Embedding dim | 128 (CLS token) |
| Index size | 31,499 training embeddings |
| Similarity | Cosine (L2-norm + dot product) |
| k | 3 |

### Decoder (Part C)
| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 3 |
| d_ff | 256 |
| dropout | 0.1 |
| max_input_len | 256 |
| max_target_len | 40 |
| Attention | Causal (upper-triangular mask) |
| Generation | Nucleus sampling (top-p=0.9, temp=0.8) |
| Optimizer | AdamW, lr=3e-4, cosine schedule |
| Epochs | 8 |

---

## Dataset Statistics

| Split | Samples | Negative | Neutral | Positive |
|-------|---------|----------|---------|----------|
| Train | 31,499 | 2,804 | 2,520 | 26,175 |
| Val | 6,750 | 594 | 521 | 5,635 |
| Test | 6,751 | 611 | 604 | 5,536 |

Categories: **Beauty**, **Home & Kitchen**, **Sports & Outdoors** (15,000 reviews each)

---

## Dependencies

All standard — available in the default Colab environment:

```
torch >= 2.0
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install extras (already in notebook Cell 7):
```bash
pip install matplotlib seaborn
```

---

## Running the Full Notebook

The notebook is **self-contained and runs top to bottom**. Each part builds on the previous one. If retraining is not needed, skip the training loop cells and go directly to the load-weights cells (clearly marked in the notebook).

**Recommended Colab runtime:** GPU (T4 or better)
