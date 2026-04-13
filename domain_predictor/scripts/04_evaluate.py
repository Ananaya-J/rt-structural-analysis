#!/usr/bin/env python3
"""
04_evaluate.py
==============
Evaluate the trained domain predictor against domain_annotator_v5 structural annotations.

Takes:
  - Trained model checkpoint
  - Candidate sequences (FASTA or from PDB files)
  - domain_annotator_v5 output (CSV with domain boundaries)

Produces:
  - Per-protein boundary comparison
  - Aggregate accuracy metrics
  - Confusion analysis
  - Publication-ready summary table

Usage:
  python 04_evaluate.py --model ./models/best_model.pt \
                        --sequences ./candidates.fasta \
                        --annotations ./domain_annotations.csv \
                        --output ./evaluation_results/

Expected annotations CSV format (from domain_annotator_v5):
  structure_id, domain, start, end, ...
  or any CSV with columns for protein ID, domain name, start/end residues.
  Configure column names via --col_* arguments.
"""

import argparse
import csv
import json
import os
import sys
import pickle
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import statistics

torch = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# We also need encoding constants — redefine here for standalone use
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_IDX = 20
PAD_IDX = 21

# Map domain_annotator_v5 names → our training label names
# Adjust this mapping based on your actual domain_annotator_v5 output
ANNOTATOR_TO_LABEL = {
    # Structural annotator names → Pfam-derived training labels
    "Fingers": "RVT_1",          # fingers is part of RVT_1
    "Palm": "RVT_1",             # palm is part of RVT_1
    "Thumb": "RVT_thumb",
    "Connection": "RVT_connect",
    "RNaseH": "RNase_H",
    "RNase_H": "RNase_H",
    "Maturase": "GIIM",
    "Group2_Maturase": "GIIM",
    # Add more mappings as needed
    "fingers": "RVT_1",
    "palm": "RVT_1",
    "thumb": "RVT_thumb",
    "connection": "RVT_connect",
    "rnaseh": "RNase_H",
    "maturase": "GIIM",
}


@dataclass
class EvaluationLabelConfig:
    """Resolved label configuration used for evaluation."""
    eval_labels: set
    training_only: set
    annotation_only: set
    collapsed_map: dict


DEFAULT_SHARED_LABEL_MAP = {
    "RVT_1": "RT_core",
    "RVT_thumb": "RT_accessory",
    "RVT_connect": "RT_accessory",
    "GIIM": "RT_accessory",
    "RNase_H": "RNase_H",
    "none": "none",
}


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    if torch is None:
        raise RuntimeError("PyTorch is required to run evaluation. Install with: pip install torch")
    sys.path.insert(0, str(Path(__file__).parent))
    from model import DomainPredictor  # noqa: E402
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta = checkpoint["meta"]
    model_args = checkpoint["args"]

    model = DomainPredictor(
        vocab_size=meta["vocab_size"],
        embed_dim=model_args["embed_dim"],
        hidden_dim=model_args["hidden_dim"],
        num_layers=model_args["num_layers"],
        num_classes=meta["num_classes"],
        dropout=0,  # No dropout at inference
        use_crf=model_args["use_crf"],
        pad_idx=meta["pad_idx"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, meta


def read_fasta(fasta_path):
    """Read sequences from FASTA file."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_seq:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id and current_seq:
        sequences[current_id] = "".join(current_seq)

    return sequences


def read_sequences_from_pdbs(pdb_dir):
    """Extract sequences from PDB files (for ESMFold outputs)."""
    sequences = {}
    pdb_dir = Path(pdb_dir)

    for pdb_file in pdb_dir.glob("*.pdb"):
        seq = []
        # 3-letter to 1-letter mapping
        aa3to1 = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        }
        seen_residues = set()
        for line in open(pdb_file):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resname = line[17:20].strip()
                resnum = int(line[22:26].strip())
                chain = line[21]
                key = (chain, resnum)
                if key not in seen_residues:
                    seen_residues.add(key)
                    seq.append(aa3to1.get(resname, 'X'))

        if seq:
            # Use last 4 chars of filename (before .pdb) as short ID
            short_id = pdb_file.stem[-4:] if len(pdb_file.stem) >= 4 else pdb_file.stem
            sequences[short_id] = "".join(seq)
            # Also store full name
            sequences[pdb_file.stem] = "".join(seq)

    return sequences


def load_annotations(csv_path, col_id="structure_id", col_domain="domain",
                     col_start="start", col_end="end"):
    """
    Load domain_annotator_v5 output.
    Returns: {protein_id: [(domain_label, start, end), ...]}
    """
    annotations = defaultdict(list)

    with open(csv_path) as f:
        reader = csv.DictReader(f)

        # Check which columns exist
        if reader.fieldnames is None:
            logger.error(f"Empty CSV: {csv_path}")
            return annotations

        logger.info(f"  CSV columns: {reader.fieldnames}")

        for row in reader:
            prot_id = row.get(col_id, "").strip()
            domain = row.get(col_domain, "").strip()
            start = row.get(col_start, "")
            end = row.get(col_end, "")

            if not prot_id or not domain or not start or not end:
                continue

            # Map annotator domain name to our label space
            mapped_label = ANNOTATOR_TO_LABEL.get(domain, ANNOTATOR_TO_LABEL.get(domain.lower()))
            if mapped_label is None:
                # Unknown domain — skip or assign to none
                continue

            try:
                annotations[prot_id].append((mapped_label, int(float(start)), int(float(end))))
            except ValueError:
                continue

    logger.info(f"  Loaded annotations for {len(annotations)} proteins")
    return annotations


def load_label_map(label_map_path):
    """Load a JSON label map; returns empty map if path is not provided."""
    if not label_map_path:
        return {}

    with open(label_map_path) as f:
        label_map = json.load(f)

    if not isinstance(label_map, dict):
        raise ValueError("--label_map_json must contain a JSON object")

    return label_map


def resolve_label_config(model_labels, annotation_labels, label_mode, label_map):
    """
    Build evaluation label config and perform consistency checks.

    label_mode:
      - strict: evaluate only in the model's native label space
      - shared: collapse labels into a shared ontology with label_map/default mapping
    """
    if label_mode == "strict":
        training_only = set(model_labels) - set(annotation_labels)
        annotation_only = set(annotation_labels) - set(model_labels)
        return EvaluationLabelConfig(
            eval_labels=set(model_labels),
            training_only=training_only,
            annotation_only=annotation_only,
            collapsed_map={},
        )

    # shared mode
    collapsed_map = dict(DEFAULT_SHARED_LABEL_MAP)
    collapsed_map.update(label_map)

    collapsed_training = set(collapsed_map.get(x, x) for x in model_labels)
    collapsed_annotations = set(collapsed_map.get(x, x) for x in annotation_labels)

    training_only = collapsed_training - collapsed_annotations
    annotation_only = collapsed_annotations - collapsed_training

    return EvaluationLabelConfig(
        eval_labels=collapsed_training | collapsed_annotations,
        training_only=training_only,
        annotation_only=annotation_only,
        collapsed_map=collapsed_map,
    )


def apply_label_mode(labels, label_mode, collapsed_map):
    """Apply label mode to a residue-label list."""
    if label_mode == "strict":
        return labels
    return [collapsed_map.get(x, x) for x in labels]


def predict_sequence(model, sequence, meta, device):
    """Run model prediction on a single sequence."""
    if torch is None:
        raise RuntimeError("PyTorch is required to run evaluation. Install with: pip install torch")
    # Encode
    seq_idx = [AA_TO_IDX.get(aa, UNK_IDX) for aa in sequence]
    seq_tensor = torch.tensor([seq_idx], dtype=torch.long).to(device)
    lengths = [len(sequence)]

    with torch.no_grad():
        preds = model.predict(seq_tensor, lengths)

    if isinstance(preds[0], list):
        pred_labels = preds[0]
    else:
        pred_labels = preds[0][:len(sequence)].tolist()

    # Convert indices to label names
    idx_to_label = meta["domain_labels"]
    return [idx_to_label[i] for i in pred_labels]


def annotations_to_residue_labels(annotations, seq_length, label_to_idx):
    """Convert domain start/end annotations to per-residue label array."""
    labels = ["none"] * seq_length

    for domain_label, start, end in annotations:
        for i in range(max(0, start - 1), min(end, seq_length)):
            if labels[i] == "none":
                labels[i] = domain_label

    return labels


def compute_boundary_comparison(pred_labels, true_labels, tolerance=5):
    """
    Compare predicted vs true domain boundaries.
    Returns per-boundary analysis.
    """
    def find_boundaries(labels):
        bounds = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                bounds.append({
                    "position": i,
                    "from": labels[i-1],
                    "to": labels[i],
                })
        return bounds

    pred_bounds = find_boundaries(pred_labels)
    true_bounds = find_boundaries(true_labels)

    # Match boundaries with tolerance
    matches = []
    unmatched_true = list(range(len(true_bounds)))
    unmatched_pred = list(range(len(pred_bounds)))

    for ti, tb in enumerate(true_bounds):
        best_pi = None
        best_dist = tolerance + 1
        for pi in unmatched_pred:
            pb = pred_bounds[pi]
            dist = abs(tb["position"] - pb["position"])
            # Check if the transition type roughly matches
            transition_match = (tb["to"] == pb["to"]) or (tb["from"] == pb["from"])
            if dist <= tolerance and dist < best_dist and transition_match:
                best_dist = dist
                best_pi = pi

        if best_pi is not None:
            matches.append({
                "true_pos": tb["position"],
                "pred_pos": pred_bounds[best_pi]["position"],
                "offset": pred_bounds[best_pi]["position"] - tb["position"],
                "true_transition": f"{tb['from']}→{tb['to']}",
                "pred_transition": f"{pred_bounds[best_pi]['from']}→{pred_bounds[best_pi]['to']}",
            })
            unmatched_true.remove(ti)
            unmatched_pred.remove(best_pi)

    return {
        "matches": matches,
        "missed_true": [true_bounds[i] for i in unmatched_true],
        "false_pred": [pred_bounds[i] for i in unmatched_pred],
        "n_true": len(true_bounds),
        "n_pred": len(pred_bounds),
        "n_matched": len(matches),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best_model.pt")
    parser.add_argument("--sequences", required=True,
                        help="FASTA file or PDB directory with candidate sequences")
    parser.add_argument("--annotations", required=True,
                        help="CSV from domain_annotator_v5")
    parser.add_argument("--output", default="./evaluation_results/")
    parser.add_argument("--tolerance", type=int, default=5,
                        help="Boundary tolerance in residues")
    # CSV column names for your annotations file
    parser.add_argument("--col_id", default="structure_id")
    parser.add_argument("--col_domain", default="domain")
    parser.add_argument("--col_start", default="start")
    parser.add_argument("--col_end", default="end")
    parser.add_argument(
        "--label_mode",
        choices=["strict", "shared"],
        default="strict",
        help=(
            "strict: native model labels only; "
            "shared: collapse labels into shared ontology to avoid train/eval label mismatch"
        ),
    )
    parser.add_argument(
        "--label_map_json",
        default="",
        help="Optional JSON map for shared label mode, e.g. {\"RVT_thumb\":\"RT_accessory\"}",
    )
    parser.add_argument(
        "--fail_on_label_mismatch",
        action="store_true",
        help="Fail fast if annotation labels include classes that are not in model label space.",
    )
    args = parser.parse_args()

    global torch
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
        except ImportError:
            parser.error("PyTorch is required for evaluation. Install with: pip install torch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    model, meta = load_model(args.model, device)
    idx_to_label = meta["domain_labels"]
    label_to_idx = meta["label_to_idx"]

    # Load sequences
    logger.info("Loading sequences...")
    seq_path = Path(args.sequences)
    if seq_path.is_dir():
        sequences = read_sequences_from_pdbs(seq_path)
    else:
        sequences = read_fasta(seq_path)
    logger.info(f"  Loaded {len(sequences)} sequences")

    # Load annotations
    logger.info("Loading structural annotations...")
    annotations = load_annotations(
        args.annotations,
        col_id=args.col_id, col_domain=args.col_domain,
        col_start=args.col_start, col_end=args.col_end,
    )

    annotation_labels = set()
    for entries in annotations.values():
        for domain, _start, _end in entries:
            annotation_labels.add(domain)

    label_map = load_label_map(args.label_map_json)
    label_config = resolve_label_config(
        model_labels=set(idx_to_label),
        annotation_labels=annotation_labels | {"none"},
        label_mode=args.label_mode,
        label_map=label_map,
    )

    if args.label_mode == "strict" and label_config.annotation_only:
        logger.warning(
            "Label mismatch detected: annotation has labels unseen during training: %s",
            sorted(label_config.annotation_only),
        )
        logger.warning(
            "These labels will have near-zero recall by construction. "
            "Use --label_mode shared or retrain with expanded labels."
        )
        if args.fail_on_label_mismatch:
            logger.error("Exiting because --fail_on_label_mismatch was set.")
            sys.exit(2)
    elif args.label_mode == "shared":
        logger.info("Shared-label evaluation enabled.")
        if label_config.training_only:
            logger.info(
                "  Collapsed labels present only in training set: %s",
                sorted(label_config.training_only),
            )
        if label_config.annotation_only:
            logger.info(
                "  Collapsed labels present only in annotation set: %s",
                sorted(label_config.annotation_only),
            )

    # Find proteins with both sequence and annotation
    common_ids = set()
    for prot_id in annotations:
        if prot_id in sequences:
            common_ids.add(prot_id)
        # Also try short ID (last 4 chars)
        elif prot_id[-4:] in sequences:
            common_ids.add(prot_id)

    logger.info(f"  {len(common_ids)} proteins with both sequence and annotation")

    if not common_ids:
        logger.error("No overlapping protein IDs between sequences and annotations!")
        logger.error("  Check your --col_id setting and sequence headers.")
        logger.error(f"  Annotation IDs (first 5): {list(annotations.keys())[:5]}")
        logger.error(f"  Sequence IDs (first 5): {list(sequences.keys())[:5]}")
        sys.exit(1)

    # ── Run predictions and compare ────────────────────────────────────────
    logger.info("\nRunning predictions...")
    all_results = []
    all_boundary_offsets = []

    per_domain_tp = defaultdict(int)
    per_domain_fp = defaultdict(int)
    per_domain_fn = defaultdict(int)

    for prot_id in sorted(common_ids):
        seq = sequences.get(prot_id, sequences.get(prot_id[-4:], ""))
        if not seq:
            continue

        annots = annotations[prot_id]

        # Predict
        pred_labels = predict_sequence(model, seq, meta, device)

        # True labels from structural annotation
        true_labels = annotations_to_residue_labels(annots, len(seq), label_to_idx)

        pred_labels = apply_label_mode(pred_labels, args.label_mode, label_config.collapsed_map)
        true_labels = apply_label_mode(true_labels, args.label_mode, label_config.collapsed_map)

        # Per-residue comparison
        for pred, true in zip(pred_labels, true_labels):
            if true != "none":
                if pred == true:
                    per_domain_tp[true] += 1
                else:
                    per_domain_fn[true] += 1
                    if pred != "none":
                        per_domain_fp[pred] += 1

        # Boundary comparison
        boundary_result = compute_boundary_comparison(
            pred_labels, true_labels, tolerance=args.tolerance
        )

        for match in boundary_result["matches"]:
            all_boundary_offsets.append(match["offset"])

        result = {
            "protein_id": prot_id,
            "seq_length": len(seq),
            "n_true_boundaries": boundary_result["n_true"],
            "n_pred_boundaries": boundary_result["n_pred"],
            "n_matched": boundary_result["n_matched"],
            "boundary_matches": boundary_result["matches"],
            "missed_boundaries": [
                {"pos": b["position"], "transition": f"{b['from']}→{b['to']}"}
                for b in boundary_result["missed_true"]
            ],
            "false_boundaries": [
                {"pos": b["position"], "transition": f"{b['from']}→{b['to']}"}
                for b in boundary_result["false_pred"]
            ],
        }
        all_results.append(result)

    # ── Aggregate metrics ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)

    # Per-domain metrics
    logger.info(f"\nPer-domain accuracy (structural annotation as ground truth):")
    logger.info(f"  {'Domain':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    logger.info(f"  {'-'*60}")

    all_f1s = []
    metric_domains = sorted(label_config.eval_labels)
    for domain in metric_domains:
        if domain == "none":
            continue
        tp = per_domain_tp.get(domain, 0)
        fp = per_domain_fp.get(domain, 0)
        fn = per_domain_fn.get(domain, 0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        support = tp + fn
        if support > 0:
            all_f1s.append(f1)
        logger.info(f"  {domain:<20} {p:>10.3f} {r:>10.3f} {f1:>10.3f} {support:>10}")

    macro_f1 = (sum(all_f1s) / len(all_f1s)) if all_f1s else 0
    logger.info(f"\n  Macro F1: {macro_f1:.4f}")

    # Boundary metrics
    total_true = sum(r["n_true_boundaries"] for r in all_results)
    total_matched = sum(r["n_matched"] for r in all_results)
    total_pred = sum(r["n_pred_boundaries"] for r in all_results)

    bnd_recall = total_matched / total_true if total_true > 0 else 0
    bnd_precision = total_matched / total_pred if total_pred > 0 else 0
    bnd_f1 = 2 * bnd_precision * bnd_recall / (bnd_precision + bnd_recall) \
        if (bnd_precision + bnd_recall) > 0 else 0

    logger.info(f"\nBoundary detection (tolerance=±{args.tolerance} residues):")
    logger.info(f"  Precision: {bnd_precision:.3f}")
    logger.info(f"  Recall:    {bnd_recall:.3f}")
    logger.info(f"  F1:        {bnd_f1:.3f}")

    if all_boundary_offsets:
        logger.info(f"\nBoundary offset statistics (predicted - true):")
        mean_offset = statistics.fmean(all_boundary_offsets)
        median_offset = statistics.median(all_boundary_offsets)
        std_offset = statistics.pstdev(all_boundary_offsets) if len(all_boundary_offsets) > 1 else 0.0
        mae_offset = statistics.fmean(abs(x) for x in all_boundary_offsets)
        logger.info(f"  Mean offset:   {mean_offset:+.1f} residues")
        logger.info(f"  Median offset: {median_offset:+.1f} residues")
        logger.info(f"  Std:           {std_offset:.1f} residues")
        logger.info(f"  MAE:           {mae_offset:.1f} residues")

    # Save results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump({
            "macro_f1": macro_f1,
            "boundary_precision": bnd_precision,
            "boundary_recall": bnd_recall,
            "boundary_f1": bnd_f1,
            "tolerance": args.tolerance,
            "n_proteins": len(all_results),
            "per_protein": all_results,
        }, f, indent=2)

    # Save per-protein summary CSV
    with open(output_dir / "per_protein_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "protein_id", "seq_length", "true_boundaries", "pred_boundaries",
            "matched_boundaries", "boundary_recall", "mean_offset"
        ])
        for r in all_results:
            recall = r["n_matched"] / r["n_true_boundaries"] if r["n_true_boundaries"] > 0 else 0
            offsets = [m["offset"] for m in r["boundary_matches"]]
            mean_offset = (sum(abs(x) for x in offsets) / len(offsets)) if offsets else float("nan")
            writer.writerow([
                r["protein_id"], r["seq_length"],
                r["n_true_boundaries"], r["n_pred_boundaries"],
                r["n_matched"], f"{recall:.3f}", f"{mean_offset:.1f}",
            ])

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
