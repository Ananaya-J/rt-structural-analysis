#!/usr/bin/env python3
"""
03_train_model_esm2.py
======================
Training script for BiLSTM domain boundary predictor
using pre-computed ESM-2 3B embeddings as input features.

Key differences from 03_train_model.py:
  - Input: 2560-dim ESM-2 embeddings (instead of one-hot 64-dim)
  - No nn.Embedding layer — embeddings loaded directly from HDF5
  - Larger hidden_dim recommended (256-512) to handle richer input
  - Same BiLSTM + softmax architecture otherwise
  - Same loss, optimizer, class weights, early stopping

Expected performance gain over one-hot model:
  - GII and retron failures largely fixed (ESM-2 encodes family context)
  - Retroviral MAE: ~5.8 res (v1) → ~2-3 res (ESM-2)
  - RVT_connect and RNase_H F1: ~0.76-0.81 → ~0.90+

Usage:
  python 03_train_model_esm2.py \\
      --embeddings training_data/esm2_embeddings.h5 \\
      --dataset    training_data/dataset_augmented_v2.json \\
      --output_dir models/esm2_3b/ \\
      --epochs 50 \\
      --hidden_dim 256 \\
      --num_layers 2 \\
      --batch_size 8 \\
      --lr 1e-3

  # Monitor training
  tail -f models/esm2_3b/training.log
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DOMAIN_LABELS = ["none", "RVT_1", "RVT_thumb", "RVT_connect", "RNase_H", "GIIM"]
LABEL_TO_IDX  = {l: i for i, l in enumerate(DOMAIN_LABELS)}
NUM_CLASSES   = len(DOMAIN_LABELS)
PAD_LABEL_IDX = -100
ESM2_3B_DIM   = 2560   # hidden dim of ESM-2 3B last layer


# ── Dataset ───────────────────────────────────────────────────────────────────

class ESM2DomainDataset(Dataset):
    """
    Loads pre-computed ESM-2 embeddings + domain labels.
    Each item: (embedding_tensor, label_tensor, length, accession)
    """
    def __init__(self, entries, h5_path, esm_dim=ESM2_3B_DIM):
        self.entries = entries
        self.h5_path = h5_path
        self.esm_dim = esm_dim
        self._h5 = None  # lazy-load per worker

    def _get_h5(self):
        # HDF5 file opened lazily — one per worker process
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        acc   = entry["accession"]
        labs  = entry["labels_encoded"]  # numpy int64 array
        n     = entry["length"]

        h5 = self._get_h5()

        if acc in h5:
            emb = h5[acc]["embeddings"][:]          # (L, 2560) float32
            # Clamp to actual sequence length (some embeddings may be truncated)
            emb = emb[:n]
        else:
            # Fallback: zero embedding if accession not in HDF5
            # This handles augmented variants (shuffled/slice) whose accessions
            # differ from originals. We fall back to embedding the base accession.
            base_acc = acc.split("_shuf")[0].split("_slice")[0].split("_0")[0].split("_1")[0]
            if base_acc in h5:
                emb = h5[base_acc]["embeddings"][:][:n]
                # For shuffled variants, zero out non-domain positions
                # (consistent with what shuffled training intends)
                if "_shuf" in acc:
                    dom_mask = (labs != LABEL_TO_IDX["none"])
                    emb_shuffled = emb.copy()
                    # Non-domain positions: use mean embedding (neutral signal)
                    mean_emb = emb[dom_mask].mean(axis=0) if dom_mask.any() else np.zeros(self.esm_dim)
                    emb_shuffled[~dom_mask] = mean_emb
                    emb = emb_shuffled
            else:
                log.warning(f"  No embedding for {acc} (base: {base_acc}) — using zeros")
                emb = np.zeros((n, self.esm_dim), dtype=np.float32)

        # Ensure correct length
        if len(emb) < n:
            pad = np.zeros((n - len(emb), self.esm_dim), dtype=np.float32)
            emb = np.concatenate([emb, pad], axis=0)
        emb = emb[:n]

        return {
            "emb":       torch.tensor(emb,  dtype=torch.float32),
            "labels":    torch.tensor(labs, dtype=torch.long),
            "length":    n,
            "accession": acc,
        }


def collate_esm2(batch):
    """Pad a batch of variable-length ESM-2 embeddings."""
    # Sort by length descending (required for pack_padded_sequence)
    batch = sorted(batch, key=lambda x: -x["length"])

    embs    = [b["emb"]    for b in batch]
    labels  = [b["labels"] for b in batch]
    lengths = [b["length"] for b in batch]
    accs    = [b["accession"] for b in batch]

    embs_padded   = pad_sequence(embs,   batch_first=True)   # (B, L_max, 2560)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=PAD_LABEL_IDX)

    return {
        "emb":       embs_padded,
        "labels":    labels_padded,
        "lengths":   lengths,
        "accessions":accs,
    }


# ── Model ─────────────────────────────────────────────────────────────────────

class ESM2DomainPredictor(nn.Module):
    """
    BiLSTM domain boundary predictor with ESM-2 embeddings as input.

    Input:  (B, L, esm_dim=2560)  — pre-computed ESM-2 3B embeddings
    Output: (B, L, num_classes=6) — per-residue domain logits

    No embedding layer — raw ESM-2 features go directly into projection + LSTM.
    """
    def __init__(self, esm_dim=ESM2_3B_DIM, proj_dim=512,
                 hidden_dim=256, num_layers=2, num_classes=6, dropout=0.3):
        super().__init__()
        self.esm_dim    = esm_dim
        self.hidden_dim = hidden_dim

        # Project from 2560 → proj_dim before LSTM (reduces compute)
        self.proj = nn.Sequential(
            nn.Linear(esm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            proj_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, emb, lengths):
        """
        emb:     (B, L, 2560)
        lengths: list of int
        Returns: (B, L, num_classes) logits
        """
        # Project embeddings
        proj = self.proj(emb)                                     # (B, L, proj_dim)

        # Pack for efficient LSTM
        packed = pack_padded_sequence(proj, lengths, batch_first=True, enforce_sorted=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        return self.classifier(self.dropout(lstm_out))            # (B, L, num_classes)

    def predict(self, emb, lengths):
        logits = self.forward(emb, lengths)
        return logits.argmax(dim=-1).cpu().tolist()


# ── Training utilities ────────────────────────────────────────────────────────

def compute_class_weights(dataset_entries):
    counts = Counter()
    for e in dataset_entries:
        for l in e["labels_encoded"]:
            counts[int(l)] += 1
    total = sum(counts.values())
    weights = {}
    for idx in range(NUM_CLASSES):
        count = counts.get(idx, 1)
        weights[idx] = min(total / (NUM_CLASSES * count), 10.0)
    mean_w = np.mean(list(weights.values()))
    return {k: v/mean_w for k, v in weights.items()}


def compute_metrics(all_preds, all_labels, label_names):
    """Per-class F1 + macro F1."""
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for pred, label in zip(all_preds, all_labels):
        if label == PAD_LABEL_IDX: continue
        if pred == label: tp[label] += 1
        else:
            fp[pred]   += 1
            fn[label]  += 1

    f1s = []
    for i, name in enumerate(label_names):
        p = tp[i] / (tp[i] + fp[i] + 1e-8)
        r = tp[i] / (tp[i] + fn[i] + 1e-8)
        f = 2*p*r / (p+r+1e-8)
        f1s.append(f)

    return f1s, np.mean(f1s)


def run_epoch(model, loader, optimizer, class_weights, device, training=True):
    model.train() if training else model.eval()

    total_loss = 0; all_preds = []; all_labels = []
    weight_tensor = torch.tensor(
        [class_weights[i] for i in range(NUM_CLASSES)], dtype=torch.float32
    ).to(device)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            emb     = batch["emb"].to(device)
            labels  = batch["labels"].to(device)
            lengths = batch["lengths"]

            logits = model(emb, lengths)                            # (B, L, C)
            loss   = nn.functional.cross_entropy(
                logits.view(-1, NUM_CLASSES),
                labels.view(-1),
                weight=weight_tensor,
                ignore_index=PAD_LABEL_IDX,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy().flatten()
            lbls  = labels.cpu().numpy().flatten()
            all_preds.extend(preds.tolist())
            all_labels.extend(lbls.tolist())

    f1s, macro_f1 = compute_metrics(all_preds, all_labels, DOMAIN_LABELS)
    avg_loss = total_loss / len(loader)
    return avg_loss, macro_f1, f1s


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_split(dataset_path, val_fraction=0.15, seed=42):
    """Load dataset.json and split train/val."""
    with open(dataset_path) as f:
        data = json.load(f)

    # Filter empty
    data = [d for d in data if set(d["labels"]) != {"none"}]

    # Encode labels
    for d in data:
        d["labels_encoded"] = np.array(
            [LABEL_TO_IDX.get(l, 0) for l in d["labels"]], dtype=np.int64
        )
        d["length"] = len(d["sequence"])

    # Stratified split by domain composition
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for i, d in enumerate(data):
        key = tuple(sorted(set(d["labels"]) - {"none"}))
        groups[key].append(i)

    train_idx, val_idx = [], []
    for key, indices in groups.items():
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_fraction))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    train = [data[i] for i in train_idx]
    val   = [data[i] for i in val_idx]
    return train, val


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train BiLSTM with ESM-2 3B embeddings")
    p.add_argument("--embeddings",  required=True, help="HDF5 file from esm2_extract.py")
    p.add_argument("--dataset",     required=True, help="dataset_augmented_v2.json")
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--esm_dim",     type=int, default=2560, help="ESM-2 embedding dim")
    p.add_argument("--proj_dim",    type=int, default=512,  help="Projection dim before LSTM")
    p.add_argument("--hidden_dim",  type=int, default=256,  help="LSTM hidden dim")
    p.add_argument("--num_layers",  type=int, default=2)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--val_fraction",type=float, default=0.15)
    p.add_argument("--patience",    type=int, default=15, help="Early stopping patience")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    log.info(f"Loading dataset from {args.dataset}")
    train_data, val_data = load_and_split(args.dataset, args.val_fraction)
    log.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    class_weights = compute_class_weights(train_data)
    log.info("Class weights:")
    for label, idx in LABEL_TO_IDX.items():
        log.info(f"  {label:15s}: {class_weights[idx]:.3f}")

    # ── Datasets + loaders ────────────────────────────────────────────────
    train_ds = ESM2DomainDataset(train_data, args.embeddings, args.esm_dim)
    val_ds   = ESM2DomainDataset(val_data,   args.embeddings, args.esm_dim)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_esm2, num_workers=args.num_workers,
                              pin_memory=device.type=="cuda")
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_esm2, num_workers=args.num_workers,
                              pin_memory=device.type=="cuda")

    # ── Model ─────────────────────────────────────────────────────────────
    model = ESM2DomainPredictor(
        esm_dim=args.esm_dim, proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_classes=NUM_CLASSES, dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")
    log.info(f"  Input dim: {args.esm_dim} → proj: {args.proj_dim} → LSTM: {args.hidden_dim}×2")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_f1 = 0.0; patience_count = 0
    history = []

    log.info(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs+1):
        t0 = time.time()

        train_loss, train_f1, _ = run_epoch(
            model, train_loader, optimizer, class_weights, device, training=True)
        val_loss,   val_f1,   val_f1s = run_epoch(
            model, val_loader,   optimizer, class_weights, device, training=False)

        scheduler.step(val_f1)
        elapsed = time.time() - t0

        log.info(f"Epoch {epoch:3d}/{args.epochs} | "
                 f"Loss {train_loss:.4f}/{val_loss:.4f} | "
                 f"MacroF1 {train_f1:.4f}/{val_f1:.4f} | {elapsed:.1f}s")

        if epoch % 10 == 0:
            log.info(f"  Per-domain val F1:")
            for i, (name, f1) in enumerate(zip(DOMAIN_LABELS, val_f1s)):
                log.info(f"    {name:15s}: {f1:.4f}")

        history.append({"epoch":epoch, "train_loss":train_loss, "val_loss":val_loss,
                         "train_f1":train_f1, "val_f1":val_f1, "val_f1s":val_f1s})

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_count = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "meta": {
                    "domain_labels": DOMAIN_LABELS,
                    "label_to_idx":  LABEL_TO_IDX,
                    "num_classes":   NUM_CLASSES,
                    "esm_dim":       args.esm_dim,
                    "model_type":    "esm2_bilstm",
                },
                "epoch": epoch,
                "val_f1": val_f1,
            }, out_dir / "best_model.pt")
            log.info(f"  ★ New best model saved (MacroF1={val_f1:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    log.info(f"\nBest model: MacroF1={best_f1:.4f}")
    log.info(f"Saved to {out_dir}/best_model.pt")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
