# RT Domain Boundary Predictor

**Goal:** Train a sequence-only neural network to predict RT domain boundaries, using Pfam/InterPro annotations as training labels, and evaluate against structural annotations from `domain_annotator_v5`.

## Pipeline Overview

```
InterPro API                              Your candidates
     │                                          │
     ▼                                          ▼
┌──────────────┐                        ┌──────────────────┐
│ 01_fetch     │                        │ FASTA / PDB dir  │
│ training data│                        │ + domain_annot   │
└──────┬───────┘                        │   v5 CSV         │
       │                                └────────┬─────────┘
       ▼                                         │
┌──────────────┐                                 │
│ 02_prepare   │                                 │
│ dataset      │                                 │
└──────┬───────┘                                 │
       │                                         │
       ▼                                         ▼
┌──────────────┐    best_model.pt     ┌──────────────────┐
│ 03_train     │ ──────────────────►  │ 04_evaluate      │
│ model        │                      │ (vs structural)  │
└──────────────┘                      └──────────────────┘
```

## Setup

```bash
# On EC2 or WSL (needs network access + GPU preferred)
pip install torch numpy requests pandas scikit-learn

# Clone/copy the scripts
cd domain_predictor/scripts/
```

## Step-by-Step

### 1. Fetch training data from InterPro

```bash
python 01_fetch_training_data.py \
    --output_dir ./training_data \
    --max_proteins 5000
```

This pulls full-length protein sequences with multi-domain RT coordinates.
Each protein gets per-residue labels: `RVT_1 | RVT_thumb | RVT_connect | RNase_H | GIIM | none`.

**What it produces:**
- `dataset.json` — full labeled dataset
- `sequences.fasta` — sequences with domain coords in headers
- `residue_labels.csv` — per-residue labels (for inspection)
- `protein_summary.csv` — protein-level overview

### 2. Prepare train/val splits

```bash
python 02_prepare_dataset.py \
    --input ./training_data/dataset.json \
    --output ./training_data/processed/ \
    --val_fraction 0.15
```

Encodes sequences, computes class weights (handles `none` class imbalance), splits data.

### 3. Train model

```bash
# Without CRF (simpler, start here)
python 03_train_model.py \
    --data_dir ./training_data/processed/ \
    --output_dir ./models/ \
    --epochs 50 \
    --hidden_dim 256 \
    --batch_size 32

# With CRF (enforces valid domain ordering)
python 03_train_model.py \
    --data_dir ./training_data/processed/ \
    --output_dir ./models/ \
    --epochs 50 \
    --hidden_dim 256 \
    --use_crf
```

**Key metrics logged:**
- `MacroF1` — macro F1 over domain classes (primary metric)
- `DomAcc` — accuracy on domain residues only (excludes `none`)
- `BndF1` — boundary detection F1 (within ±5 residues)

### 4. Evaluate against structural annotations

```bash
python 04_evaluate.py \
    --model ./models/best_model.pt \
    --sequences /path/to/candidate_sequences.fasta \
    --annotations /path/to/domain_annotator_v5_output.csv \
    --output ./evaluation_results/ \
    --tolerance 5 \
    --col_id structure_id \
    --col_domain domain \
    --col_start start \
    --col_end end
```

**Adjust `--col_*` arguments** to match your actual domain_annotator_v5 CSV column names.

**CRITICAL — Domain name mapping:**
The evaluator maps structural annotation domain names to Pfam labels.
Edit the `ANNOTATOR_TO_LABEL` dict in `04_evaluate.py` to match your annotator's output.
Key mapping: **both Fingers and Palm → RVT_1** (since PF00078 covers both).

## Architecture Decisions

### Why BiLSTM over Transformer?
- Training set is likely <10K sequences — too small for transformers to shine
- BiLSTM captures local + long-range context well for this task
- Much faster to train, iterate, debug

### Why CRF is optional?
For the contiguous domain set (RVT_1 → thumb → connection → RNaseH), CRF helps
enforce valid orderings. But some architectures skip domains (GII introns have
RVT_1 + GIIM but no thumb/connection/RNaseH), so the CRF transition matrix
needs to allow skips. Start without CRF, add it if you see ordering violations.

### Why Fingers + Palm = RVT_1?
PF00078 doesn't distinguish fingers from palm — it covers both as one unit.
This is actually fine for your first test. If the model can correctly predict
the RVT_1 boundary (where fingers+palm end and thumb begins), that's already
informative. Splitting fingers/palm is the Phase 2 hard problem.

## What to look for in results

**Good signs:**
- Boundary F1 > 0.8 at ±5 residue tolerance
- Mean absolute boundary offset < 10 residues
- Per-domain F1 > 0.85 for well-represented domains

**Red flags:**
- Systematic boundary shifts (mean offset ≠ 0) → HMM and structural boundaries
  have a consistent bias that the model learned
- Low recall on RVT_connect or GIIM → too few training examples
- High `none` confusion → model is conservative, not predicting domains

## Next steps (if this works)

1. **ESM embeddings:** Replace one-hot AA with ESM-2 embeddings for richer features
2. **Finger/Palm split:** Build custom HMMs or use structural alignment to split RVT_1
3. **Family-specific models:** Train separate models per RT family
4. **Ensemble with structural:** Use predicted boundaries as priors for domain_annotator_v5
