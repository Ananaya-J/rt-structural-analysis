#!/usr/bin/env python3
"""
01b_fetch_bulk_training_data.py
===============================
Expanded fetcher: pulls THOUSANDS of proteins with multi-domain annotations.

Strategy: Instead of just 1 representative per architecture, we query UniProt
directly for proteins that have our target Pfam domains, with coordinates.

Approach:
  1. Query UniProt for proteins containing PF00078 + at least one other target domain
  2. Retrieve sequences + domain coordinates in one shot via UniProt's REST API
  3. Build per-residue labeled dataset

This is much more data-efficient than the InterPro architecture API.

Usage:
  python 01b_fetch_bulk_training_data.py --output_dir ./training_data --max_proteins 5000
"""

import argparse
import json
import os
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

TARGET_PFAM = {
    "PF00078": "RVT_1",
    "PF06817": "RVT_thumb",
    "PF06815": "RVT_connect",
    "PF00075": "RNase_H",
    "PF08388": "GIIM",
}

REQUEST_DELAY = 0.5


def fetch_uniprot_batch(query, fields, fmt="json", size=500, max_results=5000):
    """
    Fetch proteins from UniProt REST API with pagination.
    Returns list of protein entries.
    """
    all_results = []
    params = {
        "query": query,
        "fields": fields,
        "format": fmt,
        "size": min(size, max_results),
    }

    url = UNIPROT_SEARCH
    page = 0

    while url and len(all_results) < max_results:
        page += 1
        logger.info(f"    Page {page}, {len(all_results)} proteins so far...")

        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(url, params=params if page == 1 else None, timeout=60)

            if resp.status_code != 200:
                logger.error(f"    HTTP {resp.status_code}: {resp.text[:200]}")
                break

            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            all_results.extend(results)

            # Get next page link from headers
            link_header = resp.headers.get("Link", "")
            next_match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
            if next_match:
                url = next_match.group(1)
                params = None  # next URL has params built in
            else:
                break

        except Exception as e:
            logger.error(f"    Error: {e}")
            break

    return all_results[:max_results]


def parse_pfam_domains(entry):
    """
    Extract Pfam domain coordinates from a UniProt JSON entry.
    Returns list of {pfam_id, domain_name, start, end}
    """
    domains = []
    features = entry.get("features", [])

    for feat in features:
        if feat.get("type") != "Domain" and feat.get("type") != "Region":
            continue
        # Check cross-references for Pfam
        # UniProt features don't directly give Pfam IDs in the features array
        # We need to use the xrefs approach instead

    # Better approach: use cross-references
    xrefs = entry.get("uniProtKBCrossReferences", [])
    for xref in xrefs:
        if xref.get("database") == "Pfam":
            pfam_id = xref.get("id", "")
            if pfam_id in TARGET_PFAM:
                # Extract coordinates from properties
                props = {p["key"]: p["value"] for p in xref.get("properties", [])}
                match_status = props.get("MatchStatus", "")
                entry_name = props.get("EntryName", "")

                # Location is in the xref
                locations = xref.get("locations", [])
                if not locations:
                    # Try isoformId approach or other fields
                    pass

                for loc in locations:
                    start_obj = loc.get("start", {})
                    end_obj = loc.get("end", {})

                    start = start_obj.get("value") if isinstance(start_obj, dict) else start_obj
                    end = end_obj.get("value") if isinstance(end_obj, dict) else end_obj

                    if start and end:
                        domains.append({
                            "pfam_id": pfam_id,
                            "domain_name": TARGET_PFAM[pfam_id],
                            "start": int(start),
                            "end": int(end),
                        })

    return domains


def parse_pfam_from_tsv_features(features_str):
    """
    Parse domain features from UniProt TSV 'ft_domain' field.
    Format: DOMAIN start..end; /note="..."; /evidence="..."
    """
    domains = []
    if not features_str:
        return domains

    # Split multiple features
    for feat in features_str.split("DOMAIN "):
        feat = feat.strip()
        if not feat:
            continue

        # Extract coordinates
        coord_match = re.match(r"(\d+)\.\.(\d+)", feat)
        if coord_match:
            start = int(coord_match.group(1))
            end = int(coord_match.group(2))
            # Extract note for domain name
            note_match = re.search(r'/note="([^"]+)"', feat)
            name = note_match.group(1) if note_match else ""
            domains.append({"name": name, "start": start, "end": end})

    return domains


def fetch_proteins_with_pfam_coords(max_proteins=5000):
    """
    Strategy: Fetch proteins from UniProt that have PF00078, request JSON format
    which includes cross-reference locations for Pfam domains.
    """
    logger.info("Fetching proteins with PF00078 + other target domains from UniProt...")

    # Query: has PF00078 AND at least one of (PF06817 OR PF06815 OR PF00075 OR PF08388)
    # AND reviewed (Swiss-Prot) first, then unreviewed if needed
    queries = [
        # Swiss-Prot (reviewed) — higher quality
        '(xref:pfam-PF00078) AND (xref:pfam-PF06817 OR xref:pfam-PF06815 OR xref:pfam-PF00075 OR xref:pfam-PF08388) AND (reviewed:true)',
        # TrEMBL (unreviewed) — much more data
        '(xref:pfam-PF00078) AND (xref:pfam-PF06817 OR xref:pfam-PF06815 OR xref:pfam-PF00075 OR xref:pfam-PF08388) AND (reviewed:false)',
    ]

    all_proteins = []
    seen = set()

    for query in queries:
        if len(all_proteins) >= max_proteins:
            break

        remaining = max_proteins - len(all_proteins)
        logger.info(f"\n  Query: {query[:80]}...")
        logger.info(f"  Fetching up to {remaining} proteins...")

        entries = fetch_uniprot_batch(
            query=query,
            fields="accession,sequence,xref_pfam",
            fmt="json",
            size=500,
            max_results=remaining,
        )

        for entry in entries:
            acc = entry.get("primaryAccession", "")
            if acc in seen:
                continue
            seen.add(acc)

            # Get sequence
            seq_data = entry.get("sequence", {})
            sequence = seq_data.get("value", "")
            if not sequence:
                continue

            # Get Pfam domain coordinates
            domains = parse_pfam_domains(entry)

            # Filter: must have RVT_1 + at least one other
            domain_names = {d["domain_name"] for d in domains}
            if "RVT_1" not in domain_names:
                continue
            if len(domain_names) < 2:
                continue

            all_proteins.append({
                "accession": acc,
                "sequence": sequence,
                "length": len(sequence),
                "domains": domains,
            })

        logger.info(f"  Running total: {len(all_proteins)} proteins")

    return all_proteins


def fetch_via_tsv_approach(max_proteins=5000):
    """
    Alternative approach using InterPro protein API which returns coordinates directly.
    Queries InterPro for proteins matching specific domain architectures.
    """
    logger.info("Fetching via InterPro protein API...")

    INTERPRO_PROTEIN = "https://www.ebi.ac.uk/interpro/api/protein/UniProt"

    # Fetch proteins that have PF00078
    all_proteins = []
    seen = set()

    # Target architectures (ida_id from the previous run, top ones with most proteins)
    # We'll query the protein API for each Pfam entry to get coordinates
    target_combos = [
        # (pfam_entry, additional_filter_pfam) — get proteins with both
        ("PF00078", "PF06817"),  # RVT_1 + thumb
        ("PF00078", "PF06815"),  # RVT_1 + connection
        ("PF00078", "PF00075"),  # RVT_1 + RNaseH
        ("PF00078", "PF08388"),  # RVT_1 + GIIM
    ]

    for pfam1, pfam2 in target_combos:
        if len(all_proteins) >= max_proteins:
            break

        logger.info(f"\n  Fetching proteins with {pfam1} + {pfam2}...")
        url = f"{INTERPRO_PROTEIN}/entry/pfam/{pfam1}/pfam/{pfam2}"
        params = {"page_size": 200}

        pages = 0
        max_pages = max(1, (max_proteins - len(all_proteins)) // 200 + 1)

        while url and pages < max_pages and len(all_proteins) < max_proteins:
            pages += 1
            try:
                time.sleep(REQUEST_DELAY)
                resp = requests.get(url, params=params if pages == 1 else None, timeout=60)
                if resp.status_code != 200:
                    logger.warning(f"    HTTP {resp.status_code}")
                    break

                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break

                for entry in results:
                    meta = entry.get("metadata", {})
                    acc = meta.get("accession", "")
                    if acc in seen:
                        continue
                    seen.add(acc)

                    length = meta.get("length", 0)

                    # Extract domain coordinates from entries
                    domains = []
                    for ent_group in entry.get("entries", []):
                        pfam_acc = ent_group.get("accession", "")
                        if pfam_acc not in TARGET_PFAM:
                            continue

                        for prot_loc in ent_group.get("entry_protein_locations", []):
                            for fragment in prot_loc.get("fragments", []):
                                if isinstance(fragment, dict):
                                    start = fragment.get("start", 0)
                                    end = fragment.get("end", 0)
                                    if start and end:
                                        domains.append({
                                            "pfam_id": pfam_acc,
                                            "domain_name": TARGET_PFAM[pfam_acc],
                                            "start": int(start),
                                            "end": int(end),
                                        })

                    if domains and len({d["domain_name"] for d in domains}) >= 2:
                        all_proteins.append({
                            "accession": acc,
                            "length": length,
                            "domains": domains,
                            "sequence": None,  # Need to fetch separately
                        })

                url = data.get("next")
                params = None
                logger.info(f"    Page {pages}: {len(all_proteins)} total proteins")

            except Exception as e:
                logger.error(f"    Error: {e}")
                break

    return all_proteins


def fetch_sequences_batch(accessions, batch_size=200):
    """Fetch sequences from UniProt in FASTA format."""
    sequences = {}
    acc_list = list(accessions)

    for i in range(0, len(acc_list), batch_size):
        batch = acc_list[i:i + batch_size]
        logger.info(f"  Fetching sequences {i+1}-{i+len(batch)} / {len(acc_list)}...")

        query = " OR ".join([f"accession:{acc}" for acc in batch])
        url = UNIPROT_SEARCH
        params = {
            "query": query,
            "format": "fasta",
            "size": len(batch),
        }

        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(url, params=params, timeout=120)
            if resp.status_code == 200:
                current_acc = None
                current_seq = []
                for line in resp.text.strip().split("\n"):
                    if line.startswith(">"):
                        if current_acc and current_seq:
                            sequences[current_acc] = "".join(current_seq)
                        parts = line.split("|")
                        current_acc = parts[1] if len(parts) >= 2 else line.split()[0][1:]
                        current_seq = []
                    else:
                        current_seq.append(line.strip())
                if current_acc and current_seq:
                    sequences[current_acc] = "".join(current_seq)
        except Exception as e:
            logger.error(f"  Error: {e}")

    logger.info(f"  Got {len(sequences)} / {len(acc_list)} sequences")
    return sequences


def build_residue_labels(protein):
    """Build per-residue labels from domain annotations."""
    seq = protein["sequence"]
    labels = ["none"] * len(seq)

    for domain in protein["domains"]:
        start = domain["start"] - 1
        end = domain["end"]
        name = domain["domain_name"]

        for pos in range(max(0, start), min(end, len(seq))):
            if labels[pos] == "none":
                labels[pos] = name

    return labels


def save_dataset(proteins, output_dir):
    """Save labeled dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    stats = defaultdict(int)

    for prot in proteins:
        if not prot.get("sequence"):
            continue

        labels = build_residue_labels(prot)
        for label in labels:
            stats[label] += 1

        dataset.append({
            "accession": prot["accession"],
            "sequence": prot["sequence"],
            "labels": labels,
            "domains": prot["domains"],
        })

    # Save JSON
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset, f)

    # Save FASTA
    with open(output_dir / "sequences.fasta", "w") as f:
        for d in dataset:
            domain_str = ";".join(
                f"{dom['domain_name']}:{dom['start']}-{dom['end']}" for dom in d["domains"]
            )
            f.write(f">{d['accession']} domains={domain_str}\n")
            seq = d["sequence"]
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

    # Save residue CSV
    with open(output_dir / "residue_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accession", "position", "residue", "domain_label"])
        for d in dataset:
            for i, (aa, label) in enumerate(zip(d["sequence"], d["labels"])):
                writer.writerow([d["accession"], i+1, aa, label])

    # Summary
    with open(output_dir / "protein_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accession", "length", "n_domains", "domains_present", "domain_coords"])
        for d in dataset:
            domain_names = [dom["domain_name"] for dom in d["domains"]]
            domain_coords = ";".join(
                f"{dom['domain_name']}:{dom['start']}-{dom['end']}" for dom in d["domains"]
            )
            writer.writerow([
                d["accession"], len(d["sequence"]),
                len(d["domains"]), ",".join(domain_names), domain_coords,
            ])

    return dataset, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./training_data")
    parser.add_argument("--max_proteins", type=int, default=5000)
    parser.add_argument("--method", choices=["uniprot", "interpro", "both"], default="both",
                        help="Which API to use for fetching")
    args = parser.parse_args()

    all_proteins = []
    seen = set()

    if args.method in ("uniprot", "both"):
        logger.info("=" * 60)
        logger.info("Method 1: UniProt direct query")
        logger.info("=" * 60)
        uniprot_proteins = fetch_proteins_with_pfam_coords(max_proteins=args.max_proteins)
        for p in uniprot_proteins:
            if p["accession"] not in seen:
                seen.add(p["accession"])
                all_proteins.append(p)
        logger.info(f"  UniProt: {len(uniprot_proteins)} proteins")

    if args.method in ("interpro", "both") and len(all_proteins) < args.max_proteins:
        logger.info("\n" + "=" * 60)
        logger.info("Method 2: InterPro protein API")
        logger.info("=" * 60)
        interpro_proteins = fetch_via_tsv_approach(
            max_proteins=args.max_proteins - len(all_proteins)
        )

        # These need sequences fetched separately
        need_seqs = [p for p in interpro_proteins if not p.get("sequence") and p["accession"] not in seen]
        if need_seqs:
            logger.info(f"\n  Fetching sequences for {len(need_seqs)} InterPro proteins...")
            seqs = fetch_sequences_batch([p["accession"] for p in need_seqs])
            for p in need_seqs:
                if p["accession"] in seqs:
                    p["sequence"] = seqs[p["accession"]]
                    if p["accession"] not in seen:
                        seen.add(p["accession"])
                        all_proteins.append(p)

        logger.info(f"  InterPro added: {len(all_proteins) - len(uniprot_proteins if args.method == 'both' else [])} proteins")

    logger.info(f"\n  Total unique proteins: {len(all_proteins)}")
    logger.info(f"  With sequences: {sum(1 for p in all_proteins if p.get('sequence'))}")

    # Save
    logger.info("\n" + "=" * 60)
    logger.info("Saving dataset")
    logger.info("=" * 60)
    dataset, stats = save_dataset(all_proteins, args.output_dir)

    logger.info(f"\n  Proteins saved: {len(dataset)}")
    logger.info(f"  Label distribution:")
    for label, count in sorted(stats.items(), key=lambda x: -x[1]):
        logger.info(f"    {label:20s}: {count:>10,}")

    # Domain representation
    domain_counts = defaultdict(int)
    for d in dataset:
        for dom in d["domains"]:
            domain_counts[dom["domain_name"]] += 1
    logger.info(f"\n  Domain occurrences across proteins:")
    for name, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {name:20s}: {count}")


if __name__ == "__main__":
    main()
