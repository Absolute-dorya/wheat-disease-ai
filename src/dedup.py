"""
Perceptual-hash de-duplication and leakage audit.

WHY THIS MATTERS: duplicate / near-duplicate images split across train and test
inflate accuracy and are the most common reason wheat-disease papers get
rejected. Run this on Day 1 and report the result in the paper.

Usage:
    python src/dedup.py --root data/processed
    python src/dedup.py --root data/processed --threshold 6 --quarantine data/duplicates
"""
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


def split_of(path: Path, root: Path) -> str:
    """Expected layout: root/<split>/<class>/<image>"""
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) >= 3 else "unknown"


def class_of(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[1] if len(parts) >= 3 else parts[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed",
                    help="Folder laid out as <split>/<class>/<image>")
    ap.add_argument("--threshold", type=int, default=5,
                    help="Max hamming distance to call two images duplicates")
    ap.add_argument("--hash-size", type=int, default=8, help="phash size (8 -> 64-bit)")
    ap.add_argument("--quarantine", default=None,
                    help="If set, move duplicates here instead of only reporting")
    ap.add_argument("--report", default="data/dedup_report.csv")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"[x] {root} does not exist. Build the dataset first.")

    images = find_images(root)
    if not images:
        raise SystemExit(f"[x] No images found under {root}")

    print(f"[i] Hashing {len(images)} images (phash={args.hash_size}) ...")
    hashes = []
    for p in tqdm(images):
        try:
            with Image.open(p) as im:
                hashes.append((p, imagehash.phash(im.convert("RGB"), hash_size=args.hash_size)))
        except Exception as e:  # unreadable file -> record and continue
            print(f"[!] Could not hash {p}: {e}")

    # Greedy grouping: compare each hash against existing cluster representatives.
    # O(n * clusters) but clusters stay small in practice.
    clusters = []  # list of {"rep": hash, "members": [paths]}
    for path, h in hashes:
        placed = False
        for c in clusters:
            if (h - c["rep"]) <= args.threshold:  # exact phash subtraction
                c["members"].append(path)
                placed = True
                break
        if not placed:
            clusters.append({"rep": h, "members": [path]})

    dup_clusters = [c for c in clusters if len(c["members"]) > 1]

    # Which duplicates actually cross a train/test boundary? Those are the killers.
    leaks = []
    rows = []
    for c in dup_clusters:
        splits = {split_of(p, root) for p in c["members"]}
        classes = {class_of(p, root) for p in c["members"]}
        cross_split = len(splits) > 1
        cross_class = len(classes) > 1
        for p in c["members"]:
            rows.append({
                "path": str(p.relative_to(root)),
                "split": split_of(p, root),
                "class": class_of(p, root),
                "cluster_size": len(c["members"]),
                "cross_split": cross_split,
                "cross_class": cross_class,
            })
        if cross_split or cross_class:
            leaks.append(c)

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split", "class", "cluster_size",
                                          "cross_split", "cross_class"])
        w.writeheader()
        w.writerows(rows)

    print("\n===== DEDUP SUMMARY =====")
    print(f"images hashed        : {len(hashes)}")
    print(f"unique clusters      : {len(clusters)}")
    print(f"duplicate clusters   : {len(dup_clusters)}")
    print(f"images in duplicates : {sum(len(c['members']) for c in dup_clusters)}")
    print(f"LEAKAGE clusters     : {len(leaks)}  <-- report this number in the paper")
    print(f"full report          : {report}")

    if leaks:
        print("\n[!] Leakage examples (same/near-same image in different split or class):")
        for c in leaks[:5]:
            for p in c["members"]:
                print(f"    [{split_of(p, root)}/{class_of(p, root)}] {p.relative_to(root)}")
            print("    ---")

    if args.quarantine:
        q = Path(args.quarantine)
        q.mkdir(parents=True, exist_ok=True)
        moved = 0
        for c in dup_clusters:
            # keep the first member, quarantine the rest
            keep = c["members"][0]
            for p in c["members"][1:]:
                dest = q / p.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest))
                moved += 1
        print(f"\n[i] Moved {moved} duplicates to {q} (kept one per cluster).")


if __name__ == "__main__":
    main()
