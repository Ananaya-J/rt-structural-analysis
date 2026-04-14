#!/usr/bin/env python3
"""
esm2_setup_and_extract.py
=========================
One-time script to:
  1. Install ESM-2 dependencies
  2. Download ESM-2 3B model
  3. Extract per-residue embeddings for all sequences in dataset_augmented_v2.json
  4. Save to HDF5 for fast training access

Run this on Azure A100 ONCE. Takes ~2-4 hours for 2204 sequences with 3B model.
Embeddings are cached — subsequent training runs load from HDF5, no GPU needed for that.

Usage:
  # First install deps (run once)
  pip install fair-esm h5py torch torchvision

  # Then extract embeddings
  python esm2_setup_and_extract.py \\
      --dataset  training_data/dataset_augmented_v2.json \\
      --output   training_data/esm2_embeddings.h5 \\
      --model    esm2_t36_3B_UR50D \\
      --batch_size 4 \\
      --layer    36

  # Also extract for reference sequences (for evaluation)
  python esm2_setup_and_extract.py \\
      --fasta   training_data/rt_annotated_sequences.fasta \\
      --output  training_data/esm2_embeddings_ref.h5 \\
      --model   esm2_t36_3B_UR50D \\
      --layer   36

ESM-2 3B details:
  Model:      esm2_t36_3B_UR50D
  Layers:     36 transformer blocks
  Hidden dim: 2560
  Parameters: 3B
  Best layer: 36 (last) for per-residue structural/functional info
  Disk space: ~11GB for model weights
  Output:     2560-dim vector per residue

HDF5 structure:
  /accession_name/embeddings  — shape (L, 2560), float32
  /accession_name/sequence    — stored as string attribute
  /accession_name/length      — stored as int attribute
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ESM-2 3B config
ESM_MODELS = {
    "esm2_t36_3B_UR50D":    {"layers": 36, "hidden": 2560},   # 3B  — recommended
    "esm2_t33_650M_UR50D":  {"layers": 33, "hidden": 1280},   # 650M — fallback
    "esm2_t30_150M_UR50D":  {"layers": 30, "hidden": 640},    # 150M — fast test
    "esm2_t6_8M_UR50D":     {"layers": 6,  "hidden": 320},    # 8M   — debug only
}


def check_gpu():
    if not torch.cuda.is_available():
        log.warning("No GPU detected — this will be very slow on CPU for 3B model.")
        log.warning("Strongly recommend running on Azure A100.")
        return "cpu"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    if gpu_mem < 20 and "3B" in "esm2_t36_3B_UR50D":
        log.warning("3B model needs ~20GB GPU RAM. A100 (40GB) is fine. V100 (16GB) may OOM.")
    return "cuda"


def load_esm_model(model_name, device):
    try:
        import esm
    except ImportError:
        log.error("ESM not installed. Run: pip install fair-esm")
        sys.exit(1)

    log.info(f"Loading {model_name}...")
    log.info("  (First run downloads ~11GB — this may take 10-20 minutes)")

    model, alphabet = esm.pretrained.__dict__[model_name]()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    cfg = ESM_MODELS[model_name]
    log.info(f"  Loaded: {cfg['layers']} layers, {cfg['hidden']}-dim embeddings")
    return model, alphabet, batch_converter, cfg


def read_fasta(path):
    """Read FASTA into {id: sequence} dict."""
    seqs, cur_id, cur_seq = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if cur_id: seqs[cur_id] = "".join(cur_seq)
            cur_id, cur_seq = line[1:].split()[0], []
        else:
            cur_seq.append(line)
    if cur_id: seqs[cur_id] = "".join(cur_seq)
    return seqs


def read_dataset_json(path):
    """Read dataset.json into {accession: sequence} dict."""
    with open(path) as f:
        data = json.load(f)
    # Deduplicate — augmented dataset has many variants of same accession
    seqs = {}
    for entry in data:
        acc = entry["accession"]
        seq = entry["sequence"]
        if acc not in seqs:
            seqs[acc] = seq
    log.info(f"  {len(data)} total entries, {len(seqs)} unique accessions")
    return seqs


def extract_embeddings_batch(model, batch_converter, sequences, layer, device):
    """
    Extract per-residue embeddings for a batch of sequences.

    sequences: list of (accession, sequence) tuples
    Returns: dict {accession: np.array of shape (L, hidden_dim)}
    """
    # ESM expects list of (label, sequence) tuples
    # Truncate sequences longer than 4096 (ESM-2 max)
    batch_data = []
    for acc, seq in sequences:
        if len(seq) > 4096:
            log.warning(f"  {acc}: truncating {len(seq)} → 4096 residues")
            seq = seq[:4096]
        # ESM doesn't like non-standard amino acids — replace with X
        seq_clean = "".join(aa if aa in "ACDEFGHIKLMNPQRSTVWY" else "X" for aa in seq)
        batch_data.append((acc, seq_clean))

    _, _, tokens = batch_converter(batch_data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[layer], return_contacts=False)

    embeddings = results["representations"][layer]  # (B, L+2, hidden) — +2 for BOS/EOS tokens

    output = {}
    for i, (acc, seq) in enumerate(batch_data):
        # Remove BOS token (index 0) and EOS token (index -1), get actual sequence
        seq_len = len(seq)
        emb = embeddings[i, 1:seq_len+1, :].cpu().float().numpy()  # (L, hidden)
        output[acc] = emb

    return output


def extract_all(model, batch_converter, sequences, layer, batch_size, device):
    """Extract embeddings for all sequences in batches."""
    items   = list(sequences.items())
    results = {}
    n_total = len(items)

    # Sort by length descending for efficient batching (reduces padding waste)
    items.sort(key=lambda x: -len(x[1]))

    log.info(f"Extracting embeddings for {n_total} sequences...")
    log.info(f"  Batch size: {batch_size}, Layer: {layer}")

    t_start = time.time()
    for i in range(0, n_total, batch_size):
        batch  = items[i:i+batch_size]
        embs   = extract_embeddings_batch(model, batch_converter, batch, layer, device)
        results.update(embs)

        n_done = min(i+batch_size, n_total)
        elapsed = time.time() - t_start
        rate    = n_done / elapsed
        eta     = (n_total - n_done) / rate if rate > 0 else 0
        log.info(f"  {n_done}/{n_total} sequences | "
                 f"{elapsed/60:.1f} min elapsed | ETA {eta/60:.1f} min")

        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def save_to_hdf5(embeddings, output_path, sequences):
    """Save embeddings to HDF5 file."""
    try:
        import h5py
    except ImportError:
        log.error("h5py not installed. Run: pip install h5py")
        sys.exit(1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving to {out}...")
    with h5py.File(out, "w") as f:
        for acc, emb in embeddings.items():
            grp = f.create_group(acc)
            grp.create_dataset("embeddings", data=emb, compression="gzip", compression_opts=4)
            grp.attrs["sequence"] = sequences.get(acc, "")
            grp.attrs["length"]   = emb.shape[0]
            grp.attrs["dim"]      = emb.shape[1]

    size_gb = out.stat().st_size / 1e9
    log.info(f"Saved {len(embeddings)} embeddings → {out} ({size_gb:.2f} GB)")


def verify_hdf5(path, n_check=5):
    """Quick verification that HDF5 file is readable and correct."""
    import h5py
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        log.info(f"HDF5 verification: {len(keys)} accessions stored")
        for key in keys[:n_check]:
            emb = f[key]["embeddings"][:]
            seq_len = f[key].attrs["length"]
            dim     = f[key].attrs["dim"]
            log.info(f"  {key}: shape=({seq_len}, {dim}), dtype={emb.dtype}")
    return len(keys)


def main():
    p = argparse.ArgumentParser(description="Extract ESM-2 3B embeddings for RT sequences")
    # Input: either dataset.json or FASTA
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dataset", help="Path to dataset_augmented_v2.json")
    grp.add_argument("--fasta",   help="Path to FASTA file")

    p.add_argument("--output",     required=True,  help="Output HDF5 file path")
    p.add_argument("--model",      default="esm2_t36_3B_UR50D",
                   choices=list(ESM_MODELS.keys()), help="ESM-2 model variant")
    p.add_argument("--layer",      type=int, default=None,
                   help="Which transformer layer to extract (default: last layer)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Sequences per batch (reduce if OOM, default: 4)")
    p.add_argument("--resume",     action="store_true",
                   help="Skip sequences already in output HDF5")
    args = p.parse_args()

    # ── Load sequences ────────────────────────────────────────────────────
    if args.dataset:
        log.info(f"Loading sequences from {args.dataset}")
        sequences = read_dataset_json(args.dataset)
    else:
        log.info(f"Loading sequences from {args.fasta}")
        sequences = read_fasta(args.fasta)

    log.info(f"Total unique sequences: {len(sequences)}")

    # ── Resume: skip already-extracted sequences ──────────────────────────
    if args.resume and Path(args.output).exists():
        import h5py
        with h5py.File(args.output, "r") as f:
            done = set(f.keys())
        sequences = {k: v for k, v in sequences.items() if k not in done}
        log.info(f"Resuming: {len(done)} already done, {len(sequences)} remaining")

    if not sequences:
        log.info("All sequences already extracted. Nothing to do.")
        return

    # ── Setup ─────────────────────────────────────────────────────────────
    device = check_gpu()
    model, alphabet, batch_converter, cfg = load_esm_model(args.model, device)

    layer = args.layer if args.layer is not None else cfg["layers"]
    log.info(f"Extracting from layer {layer}/{cfg['layers']} (hidden_dim={cfg['hidden']})")

    # ── Extract ───────────────────────────────────────────────────────────
    t0 = time.time()
    embeddings = extract_all(model, batch_converter, sequences, layer, args.batch_size, device)
    elapsed = time.time() - t0
    log.info(f"\nExtraction complete: {len(embeddings)} sequences in {elapsed/60:.1f} minutes")

    # ── Save ──────────────────────────────────────────────────────────────
    # If resuming, need to append to existing file
    if args.resume and Path(args.output).exists():
        import h5py
        log.info("Appending to existing HDF5...")
        with h5py.File(args.output, "a") as f:
            for acc, emb in embeddings.items():
                if acc in f: continue
                grp = f.create_group(acc)
                grp.create_dataset("embeddings", data=emb, compression="gzip", compression_opts=4)
                grp.attrs["sequence"] = sequences.get(acc, "")
                grp.attrs["length"]   = emb.shape[0]
                grp.attrs["dim"]      = emb.shape[1]
        log.info(f"Appended {len(embeddings)} embeddings to {args.output}")
    else:
        save_to_hdf5(embeddings, args.output, sequences)

    # ── Verify ────────────────────────────────────────────────────────────
    n_stored = verify_hdf5(args.output)
    log.info(f"\nDone. {n_stored} sequences stored in {args.output}")
    log.info("\nNext steps:")
    log.info("  python 03_train_model_esm2.py \\")
    log.info(f"      --embeddings {args.output} \\")
    log.info("      --dataset    training_data/dataset_augmented_v2.json \\")
    log.info("      --output_dir models/esm2_3b/ \\")
    log.info("      --epochs 50 --hidden_dim 256 --batch_size 8")


if __name__ == "__main__":
    main()
