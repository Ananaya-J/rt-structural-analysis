#!/usr/bin/env python3
"""
01_fetch_training_data.py
=========================
Pull full-length protein sequences with multi-domain RT annotations from InterPro.

Strategy:
---------
1. Query InterPro domain architecture API for proteins containing PF00078 (RVT_1)
2. Filter for architectures containing our target domains
3. For each architecture, pull representative + member proteins with full coordinates
4. Fetch actual sequences from UniProt
5. Build per-residue labeled dataset

Target Pfam domains:
  PF00078  RVT_1         (fingers+palm)
  PF06817  RVT_thumb
  PF06815  RVT_connect
  PF00075  RNase_H
  PF08388  GIIM          (Group II intron maturase)

Usage:
  python 01_fetch_training_data.py --output_dir ./training_data --max_proteins 5000
"""

import argparse
import json
import os
import sys
import time
import csv
import logging
from pathlib import Path
from collections import defaultdict

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
INTERPRO_BASE = "https://www.ebi.ac.uk/interpro/api"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"

# Our target domains
TARGET_PFAM = {
    "PF00078": "RVT_1",
    "PF06817": "RVT_thumb",
    "PF06815": "RVT_connect",
    "PF00075": "RNase_H",
    "PF08388": "GIIM",
}

# We want architectures containing at least RVT_1 + one other target domain
MIN_TARGET_DOMAINS = 2

# Rate limiting
REQUEST_DELAY = 0.35  # seconds between API calls


def safe_request(url, params=None, max_retries=3):
    """Make an HTTP request with retries and rate limiting."""
    for attempt in range(max_retries):
        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 408 or resp.status_code >= 500:
                logger.warning(f"  Retry {attempt+1}/{max_retries} for {url} (HTTP {resp.status_code})")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"  HTTP {resp.status_code} for {url}")
                return None
        except requests.exceptions.Timeout:
            logger.warning(f"  Timeout, retry {attempt+1}/{max_retries}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"  Request error: {e}")
            return None
    return None


def fetch_domain_architectures(max_pages=50):
    """
    Query InterPro for all domain architectures containing PF00078.
    Returns list of architecture dicts with domain coordinates.
    """
    architectures = []
    url = f"{INTERPRO_BASE}/entry/?ida_search=PF00078"
    page = 0

    while url and page < max_pages:
        page += 1
        logger.info(f"  Fetching architecture page {page}...")
        data = safe_request(url)
        if not data:
            break

        for entry in data.get("results", []):
            # Parse which of our target domains are present
            ida_string = entry.get("ida", "")
            present_targets = []
            for pfam_id in TARGET_PFAM:
                if pfam_id in ida_string:
                    present_targets.append(pfam_id)

            if len(present_targets) >= MIN_TARGET_DOMAINS and "PF00078" in present_targets:
                rep = entry.get("representative", {})
                architectures.append({
                    "ida": ida_string,
                    "ida_id": entry.get("ida_id", ""),
                    "representative_accession": rep.get("accession", ""),
                    "representative_length": rep.get("length", 0),
                    "representative_domains": rep.get("domains", []),
                    "unique_proteins": entry.get("unique_proteins", 0),
                    "target_domains_present": present_targets,
                })

        url = data.get("next")

    logger.info(f"  Found {len(architectures)} architectures with >= {MIN_TARGET_DOMAINS} target domains")
    return architectures


def fetch_proteins_for_architecture(ida_id, max_proteins=100):
    """
    For a given domain architecture, fetch member proteins with coordinates.
    Uses the InterPro protein API filtered by architecture.
    """
    url = f"{INTERPRO_BASE}/protein/UniProt/entry/pfam/PF00078/"
    params = {
        "ida": ida_id,
        "page_size": min(max_proteins, 200),
    }

    proteins = []
    pages_fetched = 0
    max_pages = max(1, max_proteins // 200 + 1)

    while url and pages_fetched < max_pages and len(proteins) < max_proteins:
        pages_fetched += 1
        data = safe_request(url, params if pages_fetched == 1 else None)
        if not data:
            break

        for entry in data.get("results", []):
            acc = entry.get("metadata", {}).get("accession", "")
            length = entry.get("metadata", {}).get("length", 0)

            # Extract domain matches
            domain_hits = []
            for source_db in entry.get("entries", []):
                for match in source_db.get("entry_protein_locations", []):
                    for fragment_group in match.get("fragments", []):
                        # The structure varies; handle both formats
                        if isinstance(fragment_group, dict):
                            domain_hits.append({
                                "accession": source_db.get("accession", ""),
                                "name": source_db.get("name", ""),
                                "start": fragment_group.get("start", 0),
                                "end": fragment_group.get("end", 0),
                            })

            proteins.append({
                "accession": acc,
                "length": length,
                "domain_hits": domain_hits,
            })

        url = data.get("next")
        params = None  # subsequent pages use the 'next' URL directly

    return proteins[:max_proteins]


def fetch_proteins_from_representatives(architectures, max_per_arch=50):
    """
    Simpler approach: for each architecture, fetch proteins that match it.
    Uses InterPro's protein endpoint with ida filter.
    """
    all_proteins = []
    seen_accessions = set()

    for i, arch in enumerate(architectures):
        logger.info(
            f"  Architecture {i+1}/{len(architectures)}: "
            f"{arch['ida'][:80]}... ({arch['unique_proteins']} proteins)"
        )

        # Always include the representative
        rep_acc = arch["representative_accession"]
        if rep_acc and rep_acc not in seen_accessions:
            # Parse domain coords from the representative data
            domains = []
            for d in arch["representative_domains"]:
                acc = d.get("accession", "")
                if acc in TARGET_PFAM:
                    for coord in d.get("coordinates", []):
                        for frag in coord.get("fragments", []):
                            domains.append({
                                "pfam_id": acc,
                                "domain_name": TARGET_PFAM[acc],
                                "start": frag["start"],
                                "end": frag["end"],
                            })

            if domains:
                all_proteins.append({
                    "accession": rep_acc,
                    "length": arch["representative_length"],
                    "domains": domains,
                    "architecture": arch["ida"],
                })
                seen_accessions.add(rep_acc)

    logger.info(f"  Collected {len(all_proteins)} representative proteins")
    return all_proteins


def fetch_sequences_from_uniprot(accessions, batch_size=100):
    """Fetch protein sequences from UniProt in batches."""
    sequences = {}
    acc_list = list(accessions)

    for i in range(0, len(acc_list), batch_size):
        batch = acc_list[i:i + batch_size]
        logger.info(f"  Fetching sequences {i+1}-{i+len(batch)} / {len(acc_list)}...")

        # Use UniProt search API
        query = " OR ".join([f"accession:{acc}" for acc in batch])
        url = f"{UNIPROT_BASE}/search"
        params = {
            "query": query,
            "format": "fasta",
            "size": len(batch),
        }

        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                # Parse FASTA
                current_acc = None
                current_seq = []
                for line in resp.text.strip().split("\n"):
                    if line.startswith(">"):
                        if current_acc and current_seq:
                            sequences[current_acc] = "".join(current_seq)
                        # Parse accession from header: >sp|P12345|NAME or >tr|P12345|NAME
                        parts = line.split("|")
                        if len(parts) >= 2:
                            current_acc = parts[1]
                        else:
                            current_acc = line.split()[0][1:]
                        current_seq = []
                    else:
                        current_seq.append(line.strip())
                if current_acc and current_seq:
                    sequences[current_acc] = "".join(current_seq)
            else:
                logger.warning(f"  UniProt returned HTTP {resp.status_code}")
        except Exception as e:
            logger.error(f"  UniProt fetch error: {e}")

    logger.info(f"  Retrieved {len(sequences)} / {len(acc_list)} sequences")
    return sequences


def build_residue_labels(protein, sequence):
    """
    Build per-residue domain labels for a protein.

    Returns: list of labels, one per residue.
    Label is domain name if residue falls in a domain, 'none' otherwise.
    """
    seq_len = len(sequence)
    labels = ["none"] * seq_len

    for domain in protein["domains"]:
        start = domain["start"] - 1  # Convert 1-indexed to 0-indexed
        end = domain["end"]          # end is inclusive in InterPro, so this is correct for slicing
        name = domain["domain_name"]

        for pos in range(max(0, start), min(end, seq_len)):
            # If already labeled (overlap), keep the first assignment
            if labels[pos] == "none":
                labels[pos] = name

    return labels


def save_dataset(proteins, sequences, output_dir):
    """Save the labeled dataset in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Per-residue CSV ──
    csv_path = output_dir / "residue_labels.csv"
    stats = defaultdict(int)
    proteins_written = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accession", "position", "residue", "domain_label"])

        for prot in proteins:
            acc = prot["accession"]
            if acc not in sequences:
                continue

            seq = sequences[acc]
            labels = build_residue_labels(prot, seq)

            for i, (aa, label) in enumerate(zip(seq, labels)):
                writer.writerow([acc, i + 1, aa, label])
                stats[label] += 1

            proteins_written += 1

    logger.info(f"  Wrote {csv_path} ({proteins_written} proteins)")
    logger.info(f"  Label distribution: {dict(stats)}")

    # ── 2. Protein-level summary ──
    summary_path = output_dir / "protein_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "accession", "length", "architecture",
            "n_domains", "domains_present", "domain_coords"
        ])

        for prot in proteins:
            acc = prot["accession"]
            if acc not in sequences:
                continue

            domain_names = [d["domain_name"] for d in prot["domains"]]
            domain_coords = ";".join(
                f"{d['domain_name']}:{d['start']}-{d['end']}" for d in prot["domains"]
            )

            writer.writerow([
                acc, prot["length"], prot.get("architecture", ""),
                len(prot["domains"]), ",".join(domain_names), domain_coords
            ])

    # ── 3. FASTA with domain annotations in headers ──
    fasta_path = output_dir / "sequences.fasta"
    with open(fasta_path, "w") as f:
        for prot in proteins:
            acc = prot["accession"]
            if acc not in sequences:
                continue
            domain_str = ";".join(
                f"{d['domain_name']}:{d['start']}-{d['end']}" for d in prot["domains"]
            )
            f.write(f">{acc} domains={domain_str}\n")
            seq = sequences[acc]
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

    # ── 4. JSON for easy loading ──
    json_path = output_dir / "dataset.json"
    dataset = []
    for prot in proteins:
        acc = prot["accession"]
        if acc not in sequences:
            continue
        seq = sequences[acc]
        labels = build_residue_labels(prot, seq)
        dataset.append({
            "accession": acc,
            "sequence": seq,
            "labels": labels,
            "domains": prot["domains"],
            "architecture": prot.get("architecture", ""),
        })

    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=2)

    logger.info(f"  Saved {len(dataset)} labeled proteins to {output_dir}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Fetch RT domain training data from InterPro")
    parser.add_argument("--output_dir", default="./training_data", help="Output directory")
    parser.add_argument("--max_proteins", type=int, default=5000, help="Max proteins to fetch")
    parser.add_argument("--max_arch_pages", type=int, default=50, help="Max architecture pages to scan")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step 1: Fetching domain architectures from InterPro")
    logger.info("=" * 60)
    architectures = fetch_domain_architectures(max_pages=args.max_arch_pages)

    if not architectures:
        logger.error("No architectures found! Check network connectivity.")
        sys.exit(1)

    # Sort by protein count (most common architectures first)
    architectures.sort(key=lambda x: x["unique_proteins"], reverse=True)

    # Log top architectures
    logger.info("\nTop architectures:")
    for arch in architectures[:10]:
        logger.info(
            f"  {arch['unique_proteins']:>8} proteins | "
            f"domains: {', '.join(arch['target_domains_present'])} | "
            f"{arch['ida'][:60]}"
        )

    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Collecting representative proteins with coordinates")
    logger.info("=" * 60)
    proteins = fetch_proteins_from_representatives(architectures, max_per_arch=50)

    if not proteins:
        logger.error("No proteins collected!")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Fetching sequences from UniProt")
    logger.info("=" * 60)
    accessions = {p["accession"] for p in proteins}
    sequences = fetch_sequences_from_uniprot(accessions)

    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Building and saving labeled dataset")
    logger.info("=" * 60)
    dataset = save_dataset(proteins, sequences, args.output_dir)

    # ── Summary stats ──
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    lengths = [len(d["sequence"]) for d in dataset]
    if lengths:
        logger.info(f"  Total proteins:    {len(dataset)}")
        logger.info(f"  Sequence lengths:  {min(lengths)} - {max(lengths)} (median {sorted(lengths)[len(lengths)//2]})")

        # Domain coverage
        domain_counts = defaultdict(int)
        for d in dataset:
            for dom in d["domains"]:
                domain_counts[dom["domain_name"]] += 1
        logger.info(f"  Domain counts:")
        for name, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {name:20s}: {count}")


if __name__ == "__main__":
    main()
