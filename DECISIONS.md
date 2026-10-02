# Frozen Decisions

Fill this in on **Day 1 morning**. Then stop debating. Every hour spent
re-deciding scope is an hour not spent shipping, and results become
incomparable if classes change mid-project.

Defaults are pre-filled — change them only with a reason.

---

## 1. Class taxonomy (the most important decision)

Pick **one** set. Folder names must match these exactly.

- [ ] **Set A — 5 classes (recommended default)**
  `healthy`, `leaf_rust`, `stripe_rust`, `powdery_mildew`, `septoria`

- [ ] **Set B — 7 classes (stronger paper, needs more data)**
  Set A + `fusarium_head_blight`, `loose_smut`

**Chosen:** ______________________

**Reason if not Set A:** ______________________________________

---

## 2. Severity grading

- Method: colour-based affected-area heuristic (already implemented)
- [ ] Will validate against expert scores
- [ ] Will present explicitly as a heuristic — *not* a validated scale

**Buckets:** healthy/negligible (<15%) · mild (<35%) · moderate (<60%) · severe (≥60%)

**Chosen:** ______________________________________

> Do not claim this is a standard severity scale (like a proper
> disease-severity index) unless you actually validate it against expert scores.

---

## 3. Target venue

| Option | Checked? | Notes |
|---|---|---|
| arXiv preprint (Day 5, do this regardless) | ☐ | cs.CV; confirm no conflict with target journal |
| MDPI *Agriculture* | ☐ | check current APC + whether institution waives it |
| MDPI *Agronomy* | ☐ | check current APC |
| MDPI *Plants* | ☐ | check current APC |
| Conference (open CFP?) | ☐ | verify on the official CFP page |

**Chosen:** ______________________

**Current APC:** ______________________  **Institution waiver available?** ☐ Yes ☐ No

**Does it allow preprints?** ☐ Yes ☐ No ☐ Unknown

---

## 4. Team and hardware

**Team members and roles:**
| Who | Role | Hours/day available |
|---|---|---|
| | A — data & model | |
| | B — app & deployment | |
| | C — paper & writing | |

**Are you working solo?** ☐ Yes → do the tracks in sequence each day, and cut
Set B + satellite work first.

**GPU available?**
- [ ] Local GPU — model: ____________________
- [ ] Google Colab (free tier)
- [ ] Kaggle notebooks (~30 GPU-h/week free)
- [ ] CPU only → use EfficientNet-B0 @ 224px, and expect long training runs

**Chosen:** ______________________

---

## 5. Datasets

| Dataset | URL | Licence | Accessed | Classes mapped |
|---|---|---|---|---|
| | | | | |
| | | | | |

**Held-out cross-domain test set** (never tuned on): ______________________

> This one matters most. If you skip it, you have no cross-domain result, and
> then you have no novelty.

---

## 6. Fixed hyperparameters

Record these and never change them after Day 2 — otherwise your runs are not
comparable and your results table is meaningless.

| Setting | Value |
|---|---|
| Backbone 1 | `efficientnet_b0` |
| Backbone 2 | `convnext_tiny` |
| Image size | 224 |
| Batch size | 32 |
| Learning rate | 3e-4 |
| Epochs | 15 |
| Seed | 42 |
| Split ratios | 0.70 / 0.15 / 0.15 |

**Changed (with reason):** ______________________________________

---

## 7. Scope exclusions (state these in Limitations)

Confirm all of these are OUT of scope for v1:

- ☐ Live drone flight control
- ☐ Full satellite time-series ingestion at scale
- ☐ Pathogen-level lab confirmation
- ☐ Native Android/iOS app (responsive web app instead)
- ☐ Yield-loss modelling

**Anything you moved INTO scope?** ______________________________________

> Adding scope is the main way 5-day projects fail. If you add something,
> remove something else.
