# Wheat Disease Detection — Web App Starter

AI/ML leaf-image recognition for wheat diseases, delivered as a web app with a
report map and alert feed. Built to run in ~5 days — see [PLAN.md](PLAN.md) for
the full schedule, paper strategy and anti-plagiarism protocol.

> **Status: scaffold.** The code runs, but the model needs training on your
> data first. Nothing here produces a paper-quality result until you run the
> Day 1–3 steps in [PLAN.md](PLAN.md).

## What's included

| File | Purpose |
|---|---|
| [`PLAN.md`](PLAN.md) | 5-day workplan, datasets, venues, novelty, plagiarism rules |
| `src/dedup.py` | Perceptual-hash de-duplication + train/test **leakage audit** |
| `src/train.py` | Transfer-learning trainer (timm) with metrics + confusion matrix |
| `app/app.py` | Streamlit app: diagnose, Grad-CAM, severity, map + alerts |
| `requirements.txt` | Dependencies |

## Quickstart

```bash
pip install -r requirements.txt

# 1. Lay out your images as <split>/<class>/<image>.jpg
#    data/processed/train/stripe_rust/*.jpg
#    data/processed/val/stripe_rust/*.jpg
#    data/processed/test/stripe_rust/*.jpg

# 2. Audit for duplicates/leakage -- DO THIS FIRST, it's your credibility
python src/dedup.py --root data/processed --report data/dedup_report.csv

# 3. Train a baseline
python src/train.py --data data/processed --backbone efficientnet_b0 --epochs 15

# 4. Train a second family for comparison (needed for the paper)
python src/train.py --data data/processed --backbone convnext_tiny \
    --epochs 15 --strong-aug --class-weights --run-name convnext_strong

# 5. Run the app
streamlit run app/app.py
```

## Recommended class taxonomy

Pick **one** set and freeze it — changing classes mid-project destroys your
schedule and your results table.

**Set A (5 classes, safer):** `healthy`, `leaf_rust`, `stripe_rust`,
`powdery_mildew`, `septoria`

**Set B (7 classes, stronger paper):** Set A + `fusarium_head_blight`, `loose_smut`

Use these exact lowercase-underscore folder names so the advisory rules in
`app/app.py` match automatically.

## Useful flags

`src/train.py`
- `--strong-aug` — colour/style jitter; your main weapon against the lab→field domain gap
- `--class-weights` — inverse-frequency loss for imbalanced classes
- `--img-size` — 224 (fast) / 320 (better, slower)
- `--seed` — set it and record it; reproducibility is graded

`src/dedup.py`
- `--threshold 5` — max hamming distance to call two images duplicates
- `--quarantine data/duplicates` — move duplicates out instead of only reporting

## What is implemented vs. what is a placeholder

**Real:** transfer learning, augmentation, class weighting, macro-F1, confusion
matrix, Grad-CAM, ONNX-ready state dict, map with clustering and alert rules.

**Heuristic / not validated — declare this in the paper:**
- `lesion_ratio()` in `app/app.py` estimates affected leaf area with a colour
  (HSV) rule. It is **not** a validated severity scale (no standard-area-diagram
  grading, no expert annotation). Either validate it against expert scores or
  present it explicitly as a heuristic.
- The advisory text is generic extension guidance. Replace with locally
  validated recommendations before any real deployment.

## Deliberately NOT included

Native mobile app (use this responsive web app), live drone control, pathogen
lab confirmation, yield-loss modelling. These belong in *Limitations* and
*Future Work*.

## Before publishing

Read [PLAN.md](PLAN.md) §7 first. Short version:

1. **You may reuse this code and others' open-source code** — with attribution
   and licence compliance. Track it in `CREDITS.md`.
2. **You may not reuse anyone's prose, figures or tables.** Write your own
   sentences; draw your own diagrams.
3. **Run a similarity check** and keep the report — target under 10% overall.
4. **Verify every citation is real** and disclose AI assistance per the venue's
   policy.
5. Report your **cross-domain** (lab-trained → field-tested) result honestly.
   That number, not a 99% in-domain score, is what makes this publishable.
