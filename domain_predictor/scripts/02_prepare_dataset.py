#!/usr/bin/env python3
"""
02_prepare_dataset.py
=====================
Convert the fetched InterPro data into PyTorch-ready training format.

Handles:
- Sequence encoding (one-hot or ESM embedding)
- Label encoding
- Train/val split (test = your candidate set, handled in 04_evaluate.py)
- Variable-length sequence padding/batching
- Class weight computation for imbalanced labels (lots of 'none')

Usage:
  python 02_prepare_dataset.py --input ./training_data/dataset.json \
                                --output ./training_data/processed/ \
                                --encoding onehot \
                                --val_fraction 0.15
"""

import argparse
import json
import os
import sys
import logging
import pickle
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Amino acid encoding ────────────────────────────────────────────────────
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_IDX = len(AMINO_ACIDS)  # 20 = unknown
PAD_IDX = len(AMINO_ACIDS) + 1  # 21 = padding
VOCAB_SIZE = len(AMINO_ACIDS) + 2  # 22

# ── Domain labels ──────────────────────────────────────────────────────────
DOMAIN_LABELS = ["none", "RVT_1", "RVT_thumb", "RVT_connect", "RNase_H", "GIIM"]
LABEL_TO_IDX = {label: i for i, label in enumerate(DOMAIN_LABELS)}
NUM_CLASSES = len(DOMAIN_LABELS)
PAD_LABEL_IDX = -100  # PyTorch cross-entropy ignores this index


def encode_sequence(seq):
    """Encode amino acid sequence to integer indices."""
    return [AA_TO_IDX.get(aa, UNK_IDX) for aa in seq]


def encode_labels(labels):
    """Encode domain labels to integer indices."""
    encoded = []
    for label in labels:
        if label in LABEL_TO_IDX:
            encoded.append(LABEL_TO_IDX[label])
        else:
            # Unknown domain — treat as 'none'
            logger.warning(f"  Unknown label '{label}', mapping to 'none'")
            encoded.append(LABEL_TO_IDX["none"])
    return encoded


def compute_class_weights(all_labels):
    """
    Compute inverse-frequency class weights for handling label imbalance.
    'none' residues typically dominate; domains are rarer.
    """
    counts = Counter(all_labels)
    total = sum(counts.values())
    weights = {}
    for label, idx in LABEL_TO_IDX.items():
        count = counts.get(idx, 1)
        # Inverse frequency, capped
        weights[idx] = min(total / (NUM_CLASSES * count), 10.0)

    # Normalize so mean weight = 1.0
    mean_w = np.mean(list(weights.values()))
    weights = {k: v / mean_w for k, v in weights.items()}

    return weights


def compute_boundary_positions(labels):
    """
    Extract domain boundary positions from a label sequence.
    Returns list of (position, from_domain, to_domain) tuples.
    Useful for evaluation.
    """
    boundaries = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            boundaries.append((i, labels[i - 1], labels[i]))
    return boundaries


def analyze_dataset(dataset):
    """Print dataset statistics."""
    lengths = [len(d["sequence"]) for d in dataset]
    all_labels = []
    domain_spans = defaultdict(list)

    for d in dataset:
        labels = d["labels"]
        all_labels.extend(labels)

        # Compute span lengths per domain
        current_label = labels[0]
        span_start = 0
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                if current_label != "none":
                    domain_spans[current_label].append(i - span_start)
                current_label = labels[i]
                span_start = i
        if current_label != "none":
            domain_spans[current_label].append(len(labels) - span_start)

    logger.info(f"\n{'='*60}")
    logger.info("DATASET STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"  Proteins:        {len(dataset)}")
    logger.info(f"  Total residues:  {len(all_labels):,}")
    logger.info(f"  Seq lengths:     {min(lengths)}-{max(lengths)} (median {sorted(lengths)[len(lengths)//2]})")

    label_counts = Counter(all_labels)
    logger.info(f"\n  Label distribution:")
    for label in DOMAIN_LABELS:
        count = label_counts.get(label, 0)
        pct = 100 * count / len(all_labels)
        logger.info(f"    {label:20s}: {count:>10,} ({pct:5.1f}%)")

    logger.info(f"\n  Domain span lengths (residues):")
    for domain, spans in sorted(domain_spans.items()):
        spans = np.array(spans)
        logger.info(
            f"    {domain:20s}: n={len(spans):>5}, "
            f"mean={spans.mean():.0f}, std={spans.std():.0f}, "
            f"range=[{spans.min()}-{spans.max()}]"
        )


def split_dataset(dataset, val_fraction=0.15, seed=42):
    """
    Split into train/val ensuring architecture diversity in both sets.
    Stratify by which domains are present.
    """
    rng = np.random.RandomState(seed)

    # Group by domain composition
    groups = defaultdict(list)
    for i, d in enumerate(dataset):
        key = tuple(sorted(set(d["labels"]) - {"none"}))
        groups[key].append(i)

    train_idx, val_idx = [], []
    for key, indices in groups.items():
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_fraction))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train = [dataset[i] for i in train_idx]
    val = [dataset[i] for i in val_idx]

    logger.info(f"  Split: {len(train)} train, {len(val)} val")
    return train, val


def process_split(data, name):
    """Encode a split into numpy arrays."""
    encoded = []
    for d in data:
        seq_enc = encode_sequence(d["sequence"])
        lab_enc = encode_labels(d["labels"])
        boundaries = compute_boundary_positions(d["labels"])

        encoded.append({
            "accession": d["accession"],
            "sequence": d["sequence"],
            "seq_encoded": np.array(seq_enc, dtype=np.int64),
            "labels_encoded": np.array(lab_enc, dtype=np.int64),
            "length": len(d["sequence"]),
            "boundaries": boundaries,
            "domains": d.get("domains", []),
        })

    logger.info(f"  Encoded {len(encoded)} proteins for {name}")
    return encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./training_data/dataset.json")
    parser.add_argument("--output", default="./training_data/processed/")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--encoding", choices=["onehot", "integer"], default="integer",
                        help="Sequence encoding type (model handles embedding)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    logger.info(f"Loading dataset from {args.input}")
    with open(args.input) as f:
        dataset = json.load(f)

    # Filter out proteins with no target domain labels
    dataset = [d for d in dataset if set(d["labels"]) != {"none"}]
    logger.info(f"  {len(dataset)} proteins after filtering empty annotations")

    # Analyze
    analyze_dataset(dataset)

    # Split
    logger.info("\nSplitting dataset...")
    train_data, val_data = split_dataset(dataset, val_fraction=args.val_fraction)

    # Encode
    logger.info("\nEncoding sequences...")
    train_encoded = process_split(train_data, "train")
    val_encoded = process_split(val_data, "val")

    # Compute class weights from training set
    all_train_labels = []
    for d in train_encoded:
        all_train_labels.extend(d["labels_encoded"].tolist())
    class_weights = compute_class_weights(all_train_labels)

    logger.info(f"\n  Class weights:")
    for label, idx in LABEL_TO_IDX.items():
        logger.info(f"    {label:20s}: {class_weights[idx]:.3f}")

    # Save everything
    meta = {
        "domain_labels": DOMAIN_LABELS,
        "label_to_idx": LABEL_TO_IDX,
        "num_classes": NUM_CLASSES,
        "vocab_size": VOCAB_SIZE,
        "aa_to_idx": AA_TO_IDX,
        "pad_idx": PAD_IDX,
        "unk_idx": UNK_IDX,
        "pad_label_idx": PAD_LABEL_IDX,
        "class_weights": class_weights,
        "n_train": len(train_encoded),
        "n_val": len(val_encoded),
    }

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(output_dir / "train.pkl", "wb") as f:
        pickle.dump(train_encoded, f)
    with open(output_dir / "val.pkl", "wb") as f:
        pickle.dump(val_encoded, f)

    logger.info(f"\nSaved to {output_dir}:")
    logger.info(f"  meta.json    - label mappings, class weights, vocab")
    logger.info(f"  train.pkl    - {len(train_encoded)} training proteins")
    logger.info(f"  val.pkl      - {len(val_encoded)} validation proteins")


if __name__ == "__main__":
    main()
