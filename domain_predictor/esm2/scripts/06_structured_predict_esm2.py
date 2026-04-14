#!/usr/bin/env python3
"""
06_structured_predict_esm2.py
==============================
Structured Viterbi decoder for the ESM-2 BiLSTM domain boundary predictor.

Identical biological transition logic to 06_structured_predict.py but
adapted for ESM2DomainPredictor which:
  - Takes pre-computed embeddings from HDF5 (not raw sequences)
  - Has a different forward() signature: model(emb, lengths)
  - Stores meta under different keys in the checkpoint

Usage:
  # On reference set (48 RTs, for evaluation vs xlsx)
  python3 06_structured_predict_esm2.py \\
      --model       models/esm2_3b/best_model.pt \\
      --embeddings  training_data/esm2_embeddings_ref.h5 \\
      --fasta       training_data/rt_annotated_sequences.fasta \\
      --output      predictions_esm2_structured_ref/ \\
      --min_length  20

  # On full candidate set (1358 sequences)
  # First extract candidate embeddings:
  #   python3 scripts/esm2_setup_and_extract.py \\
  #       --fasta   training_data/RTs_combined_dedup.faa \\
  #       --output  training_data/esm2_embeddings_candidates.h5 \\
  #       --model   esm2_t36_3B_UR50D --batch_size 16 --layer 36
  python3 06_structured_predict_esm2.py \\
      --model       models/esm2_3b/best_model.pt \\
      --embeddings  training_data/esm2_embeddings_candidates.h5 \\
      --fasta       training_data/RTs_combined_dedup.faa \\
      --output      predictions_esm2_structured_candidates/ \\
      --min_length  20
"""

import argparse, csv, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ── Domain constants ─────────────────────────────────────────────────────────
LABEL_NAMES = ['none', 'RVT_1', 'RVT_thumb', 'RVT_connect', 'RNase_H', 'GIIM']
NONE=0; RVT1=1; THUMB=2; CONNECT=3; RNASEH=4; GIIM=5
N_LABELS = 6


# ── Model definition (must match 03_train_model_esm2.py) ─────────────────────

class ESM2DomainPredictor(nn.Module):
    def __init__(self, esm_dim=2560, proj_dim=512,
                 hidden_dim=256, num_layers=2, num_classes=6, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(esm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            proj_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def get_log_probs(self, emb, length):
        """
        emb:    (1, L, esm_dim) float32 tensor
        length: int
        Returns: (L, N_LABELS) log-probability array (numpy)
        """
        lengths = torch.tensor([length], dtype=torch.long)
        proj    = self.proj(emb)
        packed  = pack_padded_sequence(proj, lengths, batch_first=True, enforce_sorted=True)
        out, _  = self.lstm(packed)
        out, _  = pad_packed_sequence(out, batch_first=True)
        logits  = self.classifier(self.dropout(out))     # (1, L, N_LABELS)
        return F.log_softmax(logits[0], dim=-1).detach().cpu().numpy()


# ── Transition matrix ─────────────────────────────────────────────────────────

def build_transition_matrix():
    IMPOSSIBLE = -1e6
    LOW  = np.log(0.05)
    MED  = np.log(0.3)
    HIGH = np.log(0.9)
    SELF = np.log(0.99)

    T = np.full((N_LABELS, N_LABELS), IMPOSSIBLE)
    for i in range(N_LABELS): T[i, i] = SELF

    T[NONE, NONE]    = SELF;  T[NONE, RVT1]    = HIGH
    T[NONE, THUMB]   = LOW;   T[NONE, GIIM]    = LOW

    T[RVT1, RVT1]    = SELF;  T[RVT1, NONE]    = MED
    T[RVT1, THUMB]   = HIGH;  T[RVT1, GIIM]    = HIGH
    T[RVT1, CONNECT] = LOW;   T[RVT1, RNASEH]  = LOW

    T[THUMB, THUMB]   = SELF; T[THUMB, NONE]    = HIGH
    T[THUMB, CONNECT] = HIGH; T[THUMB, RNASEH]  = LOW
    T[THUMB, RVT1]    = LOW

    T[CONNECT, CONNECT] = SELF; T[CONNECT, RNASEH] = HIGH
    T[CONNECT, NONE]    = MED;  T[CONNECT, THUMB]  = LOW

    T[RNASEH, RNASEH] = SELF; T[RNASEH, NONE]    = HIGH
    T[RNASEH, CONNECT]= LOW

    T[GIIM, GIIM]  = SELF; T[GIIM, NONE]  = HIGH; T[GIIM, RVT1] = LOW

    return T


# ── Viterbi ───────────────────────────────────────────────────────────────────

def viterbi_decode(log_probs, log_trans):
    L, N = log_probs.shape

    log_start = np.full(N, -1e6)
    log_start[NONE] = np.log(0.6); log_start[RVT1]  = np.log(0.35)
    log_start[THUMB]= np.log(0.03);log_start[GIIM]  = np.log(0.02)

    log_end = np.full(N, -1e6)
    log_end[NONE]   = np.log(0.5); log_end[RNASEH] = np.log(0.25)
    log_end[GIIM]   = np.log(0.15);log_end[THUMB]  = np.log(0.08)
    log_end[CONNECT]= np.log(0.02)

    dp      = np.full((L, N), -np.inf)
    backptr = np.zeros((L, N), dtype=np.int32)
    dp[0]   = log_start + log_probs[0]

    for t in range(1, L):
        scores   = dp[t-1, :, None] + log_trans
        best_prev= scores.argmax(axis=0)
        dp[t]    = scores[best_prev, np.arange(N)] + log_probs[t]
        backptr[t]= best_prev

    terminal  = dp[-1] + log_end
    best_last = terminal.argmax()
    path      = [int(best_last)]
    for t in range(L-1, 0, -1):
        path.append(int(backptr[t, path[-1]]))
    path.reverse()
    return path


# ── IO helpers ────────────────────────────────────────────────────────────────

def read_fasta(path):
    seqs, cur_id, cur_seq = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            if cur_id: seqs[cur_id] = ''.join(cur_seq)
            cur_id, cur_seq = line[1:].split()[0], []
        else:
            cur_seq.append(line)
    if cur_id: seqs[cur_id] = ''.join(cur_seq)
    return seqs


def path_to_segments(path, min_length=20):
    if not path: return []
    segments = []
    cur_label, cur_start = path[0], 1
    for i, label in enumerate(path[1:], start=2):
        if label != cur_label:
            if LABEL_NAMES[cur_label] != 'none':
                length = (i-1) - cur_start + 1
                if length >= min_length:
                    segments.append({'domain': LABEL_NAMES[cur_label],
                                     'start': cur_start, 'end': i-1, 'length': length})
            cur_label, cur_start = label, i
    if LABEL_NAMES[cur_label] != 'none':
        length = len(path) - cur_start + 1
        if length >= min_length:
            segments.append({'domain': LABEL_NAMES[cur_label],
                             'start': cur_start, 'end': len(path), 'length': length})
    return segments


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',      required=True, help='models/esm2_3b/best_model.pt')
    p.add_argument('--embeddings', required=True, help='HDF5 embeddings file')
    p.add_argument('--fasta',      required=True, help='FASTA of sequences to predict')
    p.add_argument('--output',     required=True, help='Output directory')
    p.add_argument('--min_length', type=int, default=20)
    p.add_argument('--transition_scale', type=float, default=1.0)
    args = p.parse_args()

    import h5py

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(args.model, map_location=device, weights_only=False)
    margs  = ckpt['args']
    meta   = ckpt['meta']

    assert meta.get('model_type') == 'esm2_bilstm', \
        "This script is for ESM-2 models only. Use 06_structured_predict.py for BiLSTM v1."

    model = ESM2DomainPredictor(
        esm_dim    = margs.get('esm_dim',    2560),
        proj_dim   = margs.get('proj_dim',   512),
        hidden_dim = margs.get('hidden_dim', 256),
        num_layers = margs.get('num_layers', 2),
        num_classes= meta['num_classes'],
        dropout    = 0,   # no dropout at inference
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded: epoch {ckpt['epoch']}, val MacroF1={ckpt['val_f1']:.4f}")
    print(f"Labels: {meta['domain_labels']}")

    # ── Build transition matrix ───────────────────────────────────────────────
    log_trans = build_transition_matrix()
    if args.transition_scale != 1.0:
        diag = np.eye(N_LABELS, dtype=bool)
        log_trans[~diag] *= args.transition_scale

    # ── Load sequences + embeddings ───────────────────────────────────────────
    seqs = read_fasta(args.fasta)
    h5   = h5py.File(args.embeddings, 'r')
    print(f"Sequences: {len(seqs)}, Embeddings: {len(h5.keys())}")

    missing = [s for s in seqs if s not in h5]
    if missing:
        print(f"WARNING: {len(missing)} sequences have no embedding — will skip: {missing[:5]}")

    # ── Predict ───────────────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_segments = {}
    t0 = time.time()

    for i, (seq_id, seq) in enumerate(seqs.items()):
        if seq_id not in h5:
            all_segments[seq_id] = []
            continue

        emb_np = h5[seq_id]['embeddings'][:]           # (L, 2560)
        L      = min(len(seq), emb_np.shape[0])
        emb_t  = torch.tensor(emb_np[:L], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            log_probs = model.get_log_probs(emb_t, L)  # (L, 6)

        path     = viterbi_decode(log_probs, log_trans)
        segments = path_to_segments(path, args.min_length)
        all_segments[seq_id] = segments

        if (i+1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(seqs)} | {elapsed/60:.1f} min")

    h5.close()
    print(f"Predictions complete: {len(seqs)} sequences in {(time.time()-t0)/60:.1f} min")

    # ── Write outputs ─────────────────────────────────────────────────────────
    with open(out_dir / 'predictions_structured.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sequence_id', 'seq_length', 'domain', 'start', 'end', 'length'])
        for seq_id, segments in all_segments.items():
            seq_len = len(seqs[seq_id])
            if not segments:
                w.writerow([seq_id, seq_len, 'none', '', '', ''])
            for seg in segments:
                w.writerow([seq_id, seq_len, seg['domain'],
                            seg['start'], seg['end'], seg['length']])

    with open(out_dir / 'predictions_structured_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sequence_id', 'seq_length', 'domains_present',
                    'RVT_1_start', 'RVT_1_end',
                    'RVT_thumb_start', 'RVT_thumb_end',
                    'RVT_connect_start', 'RVT_connect_end',
                    'RNase_H_start', 'RNase_H_end',
                    'GIIM_start', 'GIIM_end'])
        for seq_id, segments in all_segments.items():
            seq_len = len(seqs[seq_id])
            dom_set = '+'.join(s['domain'] for s in segments) if segments else 'none'
            row = [seq_id, seq_len, dom_set]
            for dom in ['RVT_1', 'RVT_thumb', 'RVT_connect', 'RNase_H', 'GIIM']:
                hits = [s for s in segments if s['domain'] == dom]
                best = max(hits, key=lambda x: x['length']) if hits else None
                row += [best['start'], best['end']] if best else ['', '']
            w.writerow(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    arch_counts = Counter()
    for segments in all_segments.values():
        dom_set = frozenset(s['domain'] for s in segments)
        arch_counts['+'.join(sorted(dom_set)) if dom_set else 'none'] += 1

    print(f"\nDomain architecture distribution:")
    for arch, count in arch_counts.most_common(15):
        print(f"  {arch:50s} {count:>5}")

    total_segs = sum(len(v) for v in all_segments.values())
    print(f"\nTotal segments: {total_segs}, avg: {total_segs/len(seqs):.2f}/seq")
    print(f"\nOutputs → {out_dir}/")
    print(f"  predictions_structured.csv")
    print(f"  predictions_structured_summary.csv")


if __name__ == '__main__':
    main()
