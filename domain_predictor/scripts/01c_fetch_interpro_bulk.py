#!/usr/bin/env python3
"""
01c_fetch_interpro_bulk.py
==========================
Fetch thousands of proteins with Pfam domain coordinates from InterPro.

Strategy: For each target Pfam entry, query InterPro's protein endpoint
which returns proteins WITH their domain coordinates.

Then cross-reference: keep only proteins that have RVT_1 + at least one other target.

Usage:
  python 01c_fetch_interpro_bulk.py --output_dir ../training_data --max_proteins 3000
"""

import argparse
import json
import sys
import time
import csv
import logging
import re
from pathlib import Path
from collections import defaultdict

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INTERPRO_BASE = "https://www.ebi.ac.uk/interpro/api"

TARGET_PFAM = {
    "PF00078": "RVT_1",
    "PF06817": "RVT_thumb",
    "PF06815": "RVT_connect",
    "PF00075": "RNase_H",
    "PF08388": "GIIM",
}

REQUEST_DELAY = 0.4


def safe_get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(url, params=params, timeout=60,
                                headers={"Accept": "application/json"})
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 408 or resp.status_code >= 500:
                logger.warning(f"  HTTP {resp.status_code}, retry {attempt+1}")
                time.sleep(2 ** attempt)
            else:
                logger.warning(f"  HTTP {resp.status_code} for {url[:100]}")
                return None
        except Exception as e:
            logger.warning(f"  Error: {e}, retry {attempt+1}")
            time.sleep(2 ** attempt)
    return None


def fetch_proteins_for_pfam(pfam_id, max_proteins=2000):
    """
    Fetch proteins annotated with a specific Pfam entry from InterPro.
    Returns {accession: {length, pfam_coords: [(pfam_id, start, end), ...]}}
    """
    proteins = {}
    url = f"{INTERPRO_BASE}/protein/UniProt/entry/pfam/{pfam_id}/"
    params = {"page_size": 200, "extra_fields": "sequence"}
    pages = 0
    max_pages = max_proteins // 200 + 1

    while url and pages < max_pages and len(proteins) < max_proteins:
        pages += 1
        logger.info(f"    {pfam_id} page {pages}, {len(proteins)} proteins...")

        data = safe_get(url, params if pages == 1 else None)
        if not data:
            break

        for entry in data.get("results", []):
            meta = entry.get("metadata", {})
            acc = meta.get("accession", "")
            length = meta.get("length", 0)
            sequence = meta.get("sequence", "")  # may or may not be present

            # Extract coordinates for THIS pfam entry
            coords = []
            for ent in entry.get("entries", []):
                ent_acc = ent.get("accession", "")
                for prot_loc in ent.get("entry_protein_locations", []):
                    for frag in prot_loc.get("fragments", []):
                        if isinstance(frag, dict):
                            start = frag.get("start")
                            end = frag.get("end")
                            if start and end:
                                coords.append((ent_acc, int(start), int(end)))

            if acc and coords:
                if acc not in proteins:
                    proteins[acc] = {
                        "length": length,
                        "sequence": sequence if sequence else None,
                        "pfam_coords": [],
                    }
                proteins[acc]["pfam_coords"].extend(coords)

        url = data.get("next")
        params = None

    logger.info(f"    {pfam_id}: {len(proteins)} proteins fetched")
    return proteins


def fetch_sequences_uniprot(accessions, batch_size=100):
    """Fetch sequences from UniProt."""
    sequences = {}
    acc_list = list(accessions)

    for i in range(0, len(acc_list), batch_size):
        batch = acc_list[i:i + batch_size]
        logger.info(f"  Seqs {i+1}-{i+len(batch)} / {len(acc_list)}...")

        query = " OR ".join([f"accession:{acc}" for acc in batch])
        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={"query": query, "format": "fasta", "size": len(batch)},
                timeout=120,
            )
            if resp.status_code == 200:
                cur_acc, cur_seq = None, []
                for line in resp.text.strip().split("\n"):
                    if line.startswith(">"):
                        if cur_acc and cur_seq:
                            sequences[cur_acc] = "".join(cur_seq)
                        parts = line.split("|")
                        cur_acc = parts[1] if len(parts) >= 2 else line.split()[0][1:]
                        cur_seq = []
                    else:
                        cur_seq.append(line.strip())
                if cur_acc and cur_seq:
                    sequences[cur_acc] = "".join(cur_seq)
        except Exception as e:
            logger.error(f"  Error: {e}")

    logger.info(f"  Got {len(sequences)} / {len(acc_list)} sequences")
    return sequences


def build_residue_labels(sequence, domains):
    labels = ["none"] * len(sequence)
    for pfam_id, start, end in domains:
        name = TARGET_PFAM.get(pfam_id, None)
        if name is None:
            continue
        for pos in range(max(0, start - 1), min(end, len(sequence))):
            if labels[pos] == "none":
                labels[pos] = name
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./training_data")
    parser.add_argument("--max_proteins", type=int, default=3000)
    args = parser.parse_args()

    # ── Step 1: Fetch proteins for each target Pfam ─────────────────────
    # We fetch proteins for each Pfam entry separately,
    # then merge to find proteins with multiple target domains
    logger.info("=" * 60)
    logger.info("Step 1: Fetching proteins per Pfam domain")
    logger.info("=" * 60)

    # Protein accession → {length, pfam_coords: [(pfam_id, start, end), ...]}
    all_proteins = {}

    # Fetch RVT_1 first (required for all), then others
    pfam_order = ["PF00078", "PF06817", "PF06815", "PF00075", "PF08388"]

    for pfam_id in pfam_order:
        logger.info(f"\n  Fetching {pfam_id} ({TARGET_PFAM[pfam_id]})...")
        proteins = fetch_proteins_for_pfam(pfam_id, max_proteins=args.max_proteins)

        for acc, info in proteins.items():
            if acc not in all_proteins:
                all_proteins[acc] = {
                    "length": info["length"],
                    "sequence": info.get("sequence"),
                    "pfam_coords": [],
                }
            # Add coordinates (avoid duplicates)
            existing = set((c[0], c[1], c[2]) for c in all_proteins[acc]["pfam_coords"])
            for coord in info["pfam_coords"]:
                if coord not in existing:
                    all_proteins[acc]["pfam_coords"].append(coord)

    logger.info(f"\n  Total unique proteins across all Pfam queries: {len(all_proteins)}")

    # ── Step 2: Filter for multi-domain proteins ────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Filtering for multi-domain proteins")
    logger.info("=" * 60)

    multi_domain = {}
    for acc, info in all_proteins.items():
        target_pfams = set()
        target_coords = []
        for pfam_id, start, end in info["pfam_coords"]:
            if pfam_id in TARGET_PFAM:
                target_pfams.add(pfam_id)
                target_coords.append((pfam_id, start, end))

        if "PF00078" in target_pfams and len(target_pfams) >= 2:
            multi_domain[acc] = {
                "length": info["length"],
                "sequence": info.get("sequence"),
                "pfam_coords": target_coords,
            }

    logger.info(f"  Multi-domain proteins (RVT_1 + ≥1 other): {len(multi_domain)}")

    # Limit
    if len(multi_domain) > args.max_proteins:
        # Prioritize proteins with more target domains
        sorted_accs = sorted(
            multi_domain.keys(),
            key=lambda a: len(set(c[0] for c in multi_domain[a]["pfam_coords"])),
            reverse=True,
        )[:args.max_proteins]
        multi_domain = {a: multi_domain[a] for a in sorted_accs}
        logger.info(f"  Trimmed to {len(multi_domain)} proteins")

    # ── Step 3: Fetch missing sequences ─────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Fetching sequences")
    logger.info("=" * 60)

    need_seqs = [acc for acc, info in multi_domain.items() if not info.get("sequence")]
    logger.info(f"  {len(need_seqs)} proteins need sequences from UniProt")

    if need_seqs:
        seqs = fetch_sequences_uniprot(need_seqs)
        for acc in need_seqs:
            if acc in seqs:
                multi_domain[acc]["sequence"] = seqs[acc]

    # Filter out proteins without sequences
    has_seq = {acc: info for acc, info in multi_domain.items() if info.get("sequence")}
    logger.info(f"  Proteins with sequences: {len(has_seq)}")

    # ── Step 4: Build and save dataset ──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Building labeled dataset")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    label_stats = defaultdict(int)
    domain_counts = defaultdict(int)

    for acc, info in has_seq.items():
        seq = info["sequence"]
        labels = build_residue_labels(seq, info["pfam_coords"])

        for label in labels:
            label_stats[label] += 1

        domains = []
        for pfam_id, start, end in info["pfam_coords"]:
            if pfam_id in TARGET_PFAM:
                domains.append({
                    "pfam_id": pfam_id,
                    "domain_name": TARGET_PFAM[pfam_id],
                    "start": start,
                    "end": end,
                })
                domain_counts[TARGET_PFAM[pfam_id]] += 1

        dataset.append({
            "accession": acc,
            "sequence": seq,
            "labels": labels,
            "domains": domains,
        })

    # Save JSON
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset, f)

    # Save FASTA
    with open(output_dir / "sequences.fasta", "w") as f:
        for d in dataset:
            dom_str = ";".join(f"{dom['domain_name']}:{dom['start']}-{dom['end']}" for dom in d["domains"])
            f.write(f">{d['accession']} domains={dom_str}\n")
            for i in range(0, len(d["sequence"]), 80):
                f.write(d["sequence"][i:i+80] + "\n")

    # Save residue CSV
    with open(output_dir / "residue_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accession", "position", "residue", "domain_label"])
        for d in dataset:
            for i, (aa, lab) in enumerate(zip(d["sequence"], d["labels"])):
                writer.writerow([d["accession"], i+1, aa, lab])

    # Save protein summary
    with open(output_dir / "protein_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accession", "length", "n_domains", "domains_present", "domain_coords"])
        for d in dataset:
            names = [dom["domain_name"] for dom in d["domains"]]
            coords = ";".join(f"{dom['domain_name']}:{dom['start']}-{dom['end']}" for dom in d["domains"])
            writer.writerow([d["accession"], len(d["sequence"]), len(d["domains"]), ",".join(names), coords])

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Proteins saved: {len(dataset)}")

    lengths = [len(d["sequence"]) for d in dataset]
    if lengths:
        logger.info(f"  Seq lengths: {min(lengths)} - {max(lengths)} (median {sorted(lengths)[len(lengths)//2]})")

    logger.info(f"\n  Residue label distribution:")
    total = sum(label_stats.values())
    for label, count in sorted(label_stats.items(), key=lambda x: -x[1]):
        logger.info(f"    {label:20s}: {count:>10,} ({100*count/total:.1f}%)")

    logger.info(f"\n  Domain occurrences:")
    for name, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {name:20s}: {count}")

    logger.info(f"\n  Files saved to {output_dir}/")


if __name__ == "__main__":
    main()
