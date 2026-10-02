"""
Consolidate raw downloads into data/processed/{train,val,test}/<class>/ and
write a committed splits.csv.

KEY FEATURE: near-duplicate images are detected BEFORE splitting and are always
assigned to the SAME split. This prevents the train/test leakage that inflates
accuracy and gets papers rejected.

Usage:
    # see the plan without copying anything
    python src/build_dataset.py --raw data/raw --dry-run

    # do it
    python src/build_dataset.py --raw data/raw --out data/processed --mode copy

    # then verify with the full audit
    python src/dedup.py --root data/processed
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from shutil import copy2

import imagehash
from PIL import Image
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def normalise(name: str) -> str:
    out = name.strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def build_class_index(raw: Path, class_map: dict):
    """Find every folder that directly contains images, group by canonical class name."""
    index = defaultdict(list)
    sources = defaultdict(set)
    for d in sorted(p for p in raw.rglob("*") if p.is_dir()):
        imgs = [f for f in sorted(d.iterdir())
                if f.is_file() and f.suffix.lower() in IMG_EXT]
        if not imgs:
            continue
        key = normalise(d.name)
        canon = class_map.get(d.name) or class_map.get(key) or key
        canon = normalise(canon)
        index[canon].extend(imgs)
        sources[canon].add(str(d.relative_to(raw)))

    for canon, srcs in sources.items():
        if len(srcs) > 1:
            print(f"[i] merged {len(srcs)} folders into class '{canon}': {sorted(srcs)}")
    return index, sources


class DSU:
    """Union-find, used to merge duplicate images into single groups."""

    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster_duplicates(hashes, threshold: int, window: int):
    """Group near-duplicate images.

    Two passes:
      1. Exact hash matches -> merged (fast, exact).
      2. Near matches: representatives are sorted by numeric hash value and
         compared only within a sliding window of `window` neighbours. A single
         bit flip moves a hash by a power of two, so near-duplicates almost
         always land close together numerically -- this is a documented
         HEURISTIC, not an exhaustive search.

    Run `src/dedup.py` afterwards as the exhaustive verification gate.
    """
    n = len(hashes)
    dsu = DSU(n)

    # Pass 1: exact matches
    by_exact = defaultdict(list)
    for i, h in enumerate(hashes):
        by_exact[str(h)].append(i)
    for _, idxs in by_exact.items():
        for j in idxs[1:]:
            dsu.union(idxs[0], j)

    # Pass 2: near matches among exact-bucket representatives
    reps = [idxs[0] for idxs in by_exact.values()]
    reps.sort(key=lambda i: int(str(hashes[i]), 16))
    for a in range(len(reps)):
        for b in range(a + 1, min(a + 1 + window, len(reps))):
            i, j = reps[a], reps[b]
            if (hashes[i] - hashes[j]) <= threshold:
                dsu.union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)
    return list(groups.values())


def assign_splits(class_groups, ratios, seed: int):
    """Assign whole duplicate-groups to splits, keeping class ratios balanced.

    Greedy: largest groups first, each dropped into the split that is furthest
    below its target size. Guarantees no group is ever split across splits.
    """
    rng = random.Random(seed)
    assignment = {}  # class -> split -> [group]
    for cls, groups in sorted(class_groups.items()):
        total = sum(len(g) for g in groups)
        target = {s: ratios[i] * total for i, s in enumerate(SPLITS)}
        got = {s: 0 for s in SPLITS}
        ordered = list(groups)
        rng.shuffle(ordered)
        ordered.sort(key=lambda g: -len(g))

        out = {s: [] for s in SPLITS}
        for g in ordered:
            s = max(SPLITS, key=lambda k: target[k] - got[k])
            out[s].append(g)
            got[s] += len(g)
        assignment[cls] = out
    return assignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="Folder of raw class folders")
    ap.add_argument("--out", default="data/processed", help="Output root")
    ap.add_argument("--splits-csv", default="data/splits.csv")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--class-map", default=None,
                    help="JSON file mapping raw folder name -> canonical class name")
    ap.add_argument("--hash-size", type=int, default=8)
    ap.add_argument("--dup-threshold", type=int, default=5)
    ap.add_argument("--window", type=int, default=25,
                    help="Sliding-window size for near-duplicate search")
    ap.add_argument("--min-per-class", type=int, default=200,
                    help="Flag classes below this count")
    ap.add_argument("--mode", choices=["copy", "symlink", "move"], default="copy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"[x] --ratios must sum to 1.0 (got {sum(args.ratios)})")

    raw = Path(args.raw)
    if not raw.exists():
        raise SystemExit(f"[x] {raw} does not exist. Download your datasets into it first.")

    class_map = {}
    if args.class_map:
        class_map = json.loads(Path(args.class_map).read_text())
        print(f"[i] loaded class map with {len(class_map)} entries")

    index, sources = build_class_index(raw, class_map)
    if not index:
        raise SystemExit(f"[x] No images found under {raw}")
    classes = sorted(index)
    print(f"\n[i] found {len(classes)} classes: {classes}")

    # ---- Flatten and hash ---------------------------------------------------
    pairs = [(p, cls) for cls in classes for p in index[cls]]
    print(f"[i] hashing {len(pairs)} images (phash={args.hash_size}) ...")

    paths, labels, hashes = [], [], []
    for p, cls in tqdm(pairs):
        try:
            with Image.open(p) as im:
                h = imagehash.phash(im.convert("RGB"), hash_size=args.hash_size)
        except Exception as e:
            print(f"[!] skipping unreadable {p}: {e}")
            continue
        paths.append(p)
        labels.append(cls)
        hashes.append(h)

    groups = cluster_duplicates(hashes, args.dup_threshold, args.window)
    print(f"[i] {len(paths)} images -> {len(groups)} duplicate-groups "
          f"({len(paths) - len(groups)} duplicate images collapsed)")

    # ---- Group by class ----------------------------------------------------
    class_groups = defaultdict(list)
    for g in groups:
        cls_counts = Counter(labels[i] for i in g)
        dominant = cls_counts.most_common(1)[0][0]
        if len(cls_counts) > 1:
            print(f"[!] group of {len(g)} images spans classes {dict(cls_counts)} "
                  f"-> labelled '{dominant}'")
        class_groups[dominant].append(g)

    # ---- Split -------------------------------------------------------------
    assignment = assign_splits(class_groups, args.ratios, args.seed)

    # ---- Summary -----------------------------------------------------------
    print("\n===== SPLIT PLAN =====")
    header = f"{'class':<26}" + "".join(f"{s:>9}" for s in SPLITS) + f"{'total':>9}"
    print(header)
    print("-" * len(header))
    thin = []
    for cls in sorted(assignment):
        counts = [sum(len(g) for g in assignment[cls][s]) for s in SPLITS]
        print(f"{cls:<26}" + "".join(f"{c:>9}" for c in counts) + f"{sum(counts):>9}")
        for s, c in zip(("train",), counts):
            if c < args.min_per_class:
                thin.append((cls, c))
    tot = [sum(sum(len(g) for g in assignment[c][s]) for c in assignment) for s in SPLITS]
    print("-" * len(header))
    print(f"{'TOTAL':<26}" + "".join(f"{t:>9}" for t in tot) + f"{sum(tot):>9}")

    if thin:
        print("\n[!] Training images below --min-per-class:")
        for cls, c in thin:
            print(f"    {cls}: {c}  <-- consider merging or dropping this class")

    if args.dry_run:
        print("\n[i] --dry-run: nothing was written.")
        return

    # ---- Write files -------------------------------------------------------
    out = Path(args.out)
    if out.exists():
        print(f"\n[!] {out} already exists -- files will be added/overwritten.")
    rows = []
    for cls in sorted(assignment):
        for split in SPLITS:
            dest_dir = out / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for g in assignment[cls][split]:
                for i in g:
                    src = paths[i]
                    dest = dest_dir / src.name
                    # avoid silent name collisions from multiple source folders
                    if dest.exists() and not dest.samefile(src):
                        stem, suf = src.stem, src.suffix
                        k = 1
                        while dest.exists():
                            dest = dest_dir / f"{stem}_dup{k}{suf}"
                            k += 1
                    if args.mode == "copy":
                        copy2(src, dest)
                    elif args.mode == "symlink":
                        dest.symlink_to(src.resolve())
                    else:
                        src.replace(dest)
                    rows.append({
                        "path": str(dest.relative_to(out)),
                        "class": cls,
                        "split": split,
                        "phash": str(hashes[i]),
                        "dup_group_size": len(g),
                        "source": str(src),
                    })

    csv_path = Path(args.splits_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class", "split", "phash",
                                          "dup_group_size", "source"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[done] {len(rows)} images written to {out}")
    print(f"[done] split file -> {csv_path}   <-- COMMIT THIS TO GIT")
    print("\n[next] verify with the exhaustive audit:")
    print("       python src/dedup.py --root " + str(out))


if __name__ == "__main__":
    main()
