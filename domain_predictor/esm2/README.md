# RT Domain Predictor — ESM-2 3B

BiLSTM domain boundary predictor using frozen ESM-2 3B embeddings as input.

## Results (epoch 34, MacroF1=0.989)
- Retroviral thumb MAE: 0.2 res (vs 5.8 res one-hot)
- 18/18 retroviral RTs within ±10 res for all 4 domains
- Retron thumb MAE: 5.0 res (vs 41.7 res one-hot)

## Large files (not in git — stored on Azure)
- training_data/esm2_embeddings.h5       (8.04 GB — 2203 training sequences)
- training_data/esm2_embeddings_ref.h5   (0.22 GB — 48 reference sequences)
- training_data/esm2_embeddings_candidates.h5 (5.86 GB — 1358 candidates)
- models/esm2_3b/best_model.pt           (~18 MB checkpoint)

## Scripts
- esm2_setup_and_extract.py   — extract embeddings from FASTA/JSON → HDF5
- 03_train_model_esm2.py      — train BiLSTM head on cached embeddings
- 06_structured_predict_esm2.py — Viterbi structured decoder for ESM-2 model

## Quick start (requires Azure A100)
```bash
# Extract embeddings
python3 scripts/esm2_setup_and_extract.py \
    --fasta  sequences.fasta \
    --output embeddings.h5 \
    --model  esm2_t36_3B_UR50D --batch_size 16 --layer 36

# Predict
python3 scripts/06_structured_predict_esm2.py \
    --model      models/esm2_3b/best_model.pt \
    --embeddings embeddings.h5 \
    --fasta      sequences.fasta \
    --output     predictions/
```
