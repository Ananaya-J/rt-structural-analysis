#!/bin/bash
# run_pipeline.sh — Execute the full domain prediction pipeline
# Usage: bash run_pipeline.sh [--use_crf]

set -euo pipefail

USE_CRF=""
if [[ "${1:-}" == "--use_crf" ]]; then
    USE_CRF="--use_crf"
    echo "CRF layer ENABLED"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)/scripts"
DATA_DIR="./training_data"
MODEL_DIR="./models"

echo "Step 1: Fetching training data from InterPro"
python "$SCRIPT_DIR/01_fetch_training_data.py" --output_dir "$DATA_DIR" --max_proteins 5000

echo "Step 2: Preparing dataset"
python "$SCRIPT_DIR/02_prepare_dataset.py" --input "$DATA_DIR/dataset.json" --output "$DATA_DIR/processed/" --val_fraction 0.15

echo "Step 3: Training model"
python "$SCRIPT_DIR/03_train_model.py" --data_dir "$DATA_DIR/processed/" --output_dir "$MODEL_DIR" --epochs 50 --hidden_dim 256 --batch_size 32 --lr 0.001 $USE_CRF

echo "Pipeline complete! Model: $MODEL_DIR/best_model.pt"
echo "Run evaluation: python $SCRIPT_DIR/04_evaluate.py --model $MODEL_DIR/best_model.pt --sequences /path/to/candidates.fasta --annotations /path/to/domain_annotator_v5.csv --output ./evaluation_results/"
