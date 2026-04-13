#!/usr/bin/env python3
"""
03_train_model.py
=================
Train a BiLSTM (+ optional CRF) per-residue domain classifier for RT sequences.

Architecture:
  AA embedding → BiLSTM → Dropout → Linear → (CRF or Softmax)

The CRF layer enforces valid domain transitions:
  none → RVT_1 → RVT_thumb → RVT_connect → RNase_H (retroviral)
  none → RVT_1 → GIIM (GII intron)

This prevents biologically impossible predictions like RNase_H → RVT_1.

Usage:
  python 03_train_model.py --data_dir ./training_data/processed/ \
                           --output_dir ./models/ \
                           --epochs 50 \
                           --hidden_dim 256 \
                           --use_crf
"""

import argparse
import json
import os
import sys
import pickle
import logging
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
except ImportError:
    print("pip install torch")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Import model components from shared module
sys.path.insert(0, str(Path(__file__).parent))
from model import DomainPredictor, DomainDataset, CRF, collate_fn  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(model, dataloader, device, idx_to_label):
    """Evaluate model and return per-domain metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_lengths = []

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["seq"].to(device)
            labels = batch["labels"]
            lengths = batch["lengths"]

            preds = model.predict(seq, lengths)

            for i, l in enumerate(lengths):
                if isinstance(preds[i], list):
                    p = preds[i][:l]
                else:
                    p = preds[i][:l]
                    if isinstance(p, torch.Tensor):
                        p = p.tolist()

                t = labels[i][:l].tolist()
                all_preds.extend(p)
                all_labels.extend(t)

    # Per-class metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    results = {}
    for idx, label in enumerate(idx_to_label):
        pred_mask = all_preds == idx
        true_mask = all_labels == idx

        tp = (pred_mask & true_mask).sum()
        fp = (pred_mask & ~true_mask).sum()
        fn = (~pred_mask & true_mask).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(true_mask.sum()),
        }

    # Overall accuracy
    valid = all_labels != -100
    accuracy = (all_preds[valid] == all_labels[valid]).mean()
    results["overall_accuracy"] = float(accuracy)

    # Accuracy excluding 'none' (domain-only accuracy)
    domain_mask = (all_labels != 0) & (all_labels != -100)
    if domain_mask.sum() > 0:
        domain_accuracy = (all_preds[domain_mask] == all_labels[domain_mask]).mean()
        results["domain_accuracy"] = float(domain_accuracy)

    return results


def compute_boundary_accuracy(model, dataloader, device, tolerance=5):
    """
    Evaluate boundary prediction accuracy.
    A predicted boundary is 'correct' if it's within ±tolerance residues
    of a true boundary.
    """
    model.eval()
    total_true_boundaries = 0
    matched_boundaries = 0
    false_positive_boundaries = 0

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["seq"].to(device)
            labels = batch["labels"]
            lengths = batch["lengths"]

            preds = model.predict(seq, lengths)

            for i, l in enumerate(lengths):
                if isinstance(preds[i], list):
                    pred_seq = preds[i][:l]
                else:
                    pred_seq = preds[i][:l]
                    if isinstance(pred_seq, torch.Tensor):
                        pred_seq = pred_seq.tolist()

                true_seq = labels[i][:l].tolist()

                # Find boundaries (transitions between different labels)
                true_bounds = set()
                pred_bounds = set()
                for j in range(1, l):
                    if true_seq[j] != true_seq[j-1]:
                        true_bounds.add(j)
                    if pred_seq[j] != pred_seq[j-1]:
                        pred_bounds.add(j)

                # Match with tolerance
                matched_true = set()
                matched_pred = set()
                for tb in true_bounds:
                    for pb in pred_bounds:
                        if abs(tb - pb) <= tolerance and pb not in matched_pred:
                            matched_true.add(tb)
                            matched_pred.add(pb)
                            break

                total_true_boundaries += len(true_bounds)
                matched_boundaries += len(matched_true)
                false_positive_boundaries += len(pred_bounds) - len(matched_pred)

    recall = matched_boundaries / total_true_boundaries if total_true_boundaries > 0 else 0
    precision = matched_boundaries / (matched_boundaries + false_positive_boundaries) \
        if (matched_boundaries + false_positive_boundaries) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "tolerance": tolerance,
        "total_true_boundaries": total_true_boundaries,
        "matched": matched_boundaries,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    with open(Path(args.data_dir) / "meta.json") as f:
        meta = json.load(f)
    with open(Path(args.data_dir) / "train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(Path(args.data_dir) / "val.pkl", "rb") as f:
        val_data = pickle.load(f)

    idx_to_label = meta["domain_labels"]
    logger.info(f"Labels: {idx_to_label}")
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # DataLoaders
    train_dataset = DomainDataset(train_data)
    val_dataset = DomainDataset(val_data)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    # Model
    model = DomainPredictor(
        vocab_size=meta["vocab_size"],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=meta["num_classes"],
        dropout=args.dropout,
        use_crf=args.use_crf,
        pad_idx=meta["pad_idx"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"CRF: {'enabled' if args.use_crf else 'disabled'}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Training
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        t0 = time.time()
        for batch in train_loader:
            seq = batch["seq"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"]

            optimizer.zero_grad()
            loss = model(seq, labels, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0

        # Evaluate
        val_results = evaluate(model, val_loader, device, idx_to_label)
        boundary_results = compute_boundary_accuracy(model, val_loader, device, tolerance=5)

        # Compute macro F1 over domain classes (excluding 'none')
        domain_f1s = [
            val_results[label]["f1"]
            for label in idx_to_label
            if label != "none" and val_results[label]["support"] > 0
        ]
        macro_f1 = np.mean(domain_f1s) if domain_f1s else 0

        scheduler.step(macro_f1)

        epoch_log = {
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": val_results["overall_accuracy"],
            "domain_accuracy": val_results.get("domain_accuracy", 0),
            "macro_f1": macro_f1,
            "boundary_f1": boundary_results["boundary_f1"],
            "per_domain": {k: v for k, v in val_results.items() if isinstance(v, dict)},
        }
        history.append(epoch_log)

        # Log
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss {avg_loss:.4f} | "
            f"Acc {val_results['overall_accuracy']:.3f} | "
            f"DomAcc {val_results.get('domain_accuracy', 0):.3f} | "
            f"MacroF1 {macro_f1:.3f} | "
            f"BndF1 {boundary_results['boundary_f1']:.3f} | "
            f"{elapsed:.1f}s"
        )

        if epoch % 10 == 0 or epoch == 1:
            for label in idx_to_label:
                if label in val_results and isinstance(val_results[label], dict):
                    r = val_results[label]
                    logger.info(
                        f"    {label:20s}: P={r['precision']:.3f} R={r['recall']:.3f} "
                        f"F1={r['f1']:.3f} (n={r['support']})"
                    )

        # Save best
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "meta": meta,
                "args": vars(args),
                "val_results": val_results,
                "boundary_results": boundary_results,
                "macro_f1": macro_f1,
            }, output_dir / "best_model.pt")
            logger.info(f"    ★ New best model saved (F1={macro_f1:.4f})")

        # Early stopping
        if epoch - best_epoch > args.patience:
            logger.info(f"  Early stopping after {args.patience} epochs without improvement")
            break

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nBest model: epoch {best_epoch}, macro F1 = {best_f1:.4f}")
    logger.info(f"Saved to {output_dir / 'best_model.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./training_data/processed/")
    parser.add_argument("--output_dir", default="./models/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--use_crf", action="store_true", default=False,
                        help="Use CRF layer for structured prediction")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
