# Wheat Disease Detection — 5-Day Build & Paper Plan

**Problem statement:** Wheat is affected by fungal (rusts, smuts, powdery mildew, root/head blights), bacterial (Black Chaff) and viral (Barley Yellow Dwarf Virus) diseases causing leaf spots, wilting and ear/grain blighting → major yield loss. Rusts, blight and Karnal bunt spread fast.

**Expected solution:** AI/ML image recognition + drone/satellite data + mobile/web delivery, giving farmers real-time detection, mapping and alerts.

---

## 0. Reality check (read this first)

Two honest statements that shape everything below:

1. **You cannot get a paper *accepted* in 5 days.** Peer review is not under your control. Realistic fast-track timelines (third-party publisher statistics, treat as indicative):
   - MDPI *Agriculture* — first decision roughly ~18–19 days
   - MDPI *IJ Plant Biology* — first decision roughly ~17.5 days
   - MDPI *Agronomy* — average review roughly ~33 days
   - Most IEEE/Springer conferences — review 4–12 weeks

   **What you CAN do in 5 days:** have a *submittable* manuscript + a *working, deployed, demoable* app + an arXiv preprint (which is citable immediately and timestamps your priority). That is a genuinely strong 5-day outcome.

2. **"Reuse everything available" = code and data, NOT prose, figures, or tables.** Reusing another paper's text or images is the #1 cause of plagiarism flags. Reusing another project's *code* (with attribution + license compliance) is normal and expected. Section 7 is the rulebook.

**Your realistic 5-day definition of done:**
- Working web app, publicly reachable, classifying uploaded wheat images + showing a map/alert view
- Trained model with an honest evaluation table (accuracy, F1, confusion matrix, cross-dataset test)
- Manuscript draft complete in the target template, similarity-checked, figures original
- arXiv preprint posted; journal submission prepared and sent Day 5 evening

---

## 1. Locked scope (do not expand this)

Scope creep is what kills 5-day projects. Freeze this now.

**In scope**
| Item | Decision |
|---|---|
| Crop | Wheat only |
| Task 1 | Single-image **classification** into disease classes + healthy |
| Task 2 | **Severity grading** (configurable bucket: healthy / mild / moderate / severe) |
| Task 3 | **Interactive map + alert feed** (report pins, hotspot clustering, advisory text) |
| Task 4 | Cross-domain evaluation (field-like images ≠ lab images) — *this is your novelty hook* |
| Input | Mobile photo upload, drone/UAV tile upload, batch upload |
| Output | Class + confidence + severity + advisory + geotagged pin on map |
| Deliverable | Web app (responsive, works on phone browser) + paper |

**Explicitly out of scope for v1** (list in the paper's *Limitations*, which also protects you from reviewer attacks)
- Live drone flight integration / autopilot
- Real satellite time-series ingestion at scale
- Species-level pathogen confirmation (needs lab validation)
- Native Android/iOS app — a responsive PWA instead

**Choose your classes (pick ONE set, don't mix):**
- Set A (5 classes): Healthy, Leaf Rust, Stripe/Yellow Rust, Powdery Mildew, Septoria Leaf Blotch
- Set B (7 classes): Set A + Fusarium Head Blight, Loose Smut

Set A is safer if your dataset is thin. Set B is more impressive for the paper. Decide on Day 1 morning and never change it.

---

## 2. Architecture

```
                    ┌──────────────────────────────────────────┐
   Farmer phone ───▶│  Web app (Streamlit or FastAPI + React)  │
   (photo upload)   │  ├─ Predict tab                          │
                    │  ├─ Severity grading                     │
                    │  ├─ Map + alert feed                     │
                    │  └─ Batch (drone tiles)                  │
                    └───────────────┬──────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
                    │  Inference API                           │
                    │  ├─ Preprocess (resize, CLAHE, normalize)│
                    │  ├─ Classifier (fine-tuned CNN/ViT)      │
                    │  ├─ Grad-CAM explainability              │
                    │  └─ Advisory rule engine (class→action)  │
                    └───────────────┬──────────────────────────┘
                                    │
      ┌─────────────────────────────┼─────────────────────────────┐
      │                             │                             │
┌─────▼──────┐            ┌─────────▼─────────┐        ┌──────────▼─────────┐
│ Model      │            │ Data layer        │        │ Geo layer          │
│ transfer   │            │ ├─ image datasets │        │ ├─ GPS from EXIF   │
│ learning   │            │ ├─ augmentation   │        │ ├─ PostGIS/SQLite  │
│ (timm)     │            │ └─ severity labels│        │ ├─ Folium/Leaflet  │
└────────────┘            └───────────────────┘        │ └─ hotspot cluster │
                                                       └────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │ Remote sensing (bonus)   │
                    │ Sentinel-2 via GEE       │
                    │ NDVI + rust indices      │
                    └──────────────────────────┘
```

**Storage:** SQLite is fine for 5 days. Do not spend a day on Postgres.

---

## 3. Datasets (verified starting points)

Use **at least two sources**. Multi-source = cross-dataset evaluation = your main scientific contribution. Searching a single Kaggle set and reporting 99% is no longer publishable on its own.

| Dataset | Content | Use | Link |
|---|---|---|---|
| Kaggle **"Wheat Plant Diseases"** | Healthy + common wheat diseases (leaf rust etc.) | Primary training | Verified to exist; search Kaggle for `Wheat Plant Diseases`. A SPIE proceedings paper uses it as its baseline. |
| **WheatRust21** | ~5,325 images, 7 fungal diseases (Blast, Brown Rust, Stripe Rust, Fusarium Head Blight, Loose Smut, Powdery Mildew, ...) | Primary training | Described in [ResearchGate dataset description](https://www.researchgate.net/figure/Description-of-WheatRust21-Image-Dataset_tbl1_381064091) — contact authors / check paper supplementary for access |
| **NWRD** (North West Rust Dataset) | ~100 high-res UAV wheat-rust images (native ~4000x6016) | Drone/field-domain test set | [ResearchGate dataset figure](https://www.researchgate.net/figure/Sample-images-from-the-NWRD-dataset-annotated-images-showing-rust-disease-along-with_fig2_372924014) |
| Kaggle **Global Wheat Detection** | Large field wheat-head detection, deliberately includes domain shift | Head detection / domain-gap evidence | Well-known public benchmark |
| Kaggle **Beyond Visible Spectrum: AI for Agriculture 2024** | Wheat stripe rust, aphids, rice blast/planthopper | Additional rust data | [Competition page](https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2024p2/overview/abstract) |
| **WDSD** | Diverse wheat images built for vision-language diagnosis | Related work + possible external test | [Computers and Electronics in Agriculture 2024](https://dl.acm.org/doi/abs/10.1016/j.compag.2024.109587) |

**Remote sensing (drone/satellite) data — free, real, citable**
- Sentinel-2 imagery via Google Earth Engine: [Sentinel-2 catalog](https://developers.google.com/earth-engine/datasets/catalog/sentinel-2)
- Established yellow-rust spectral index from Sentinel-2: [Sensors 2018, 18(3), 868](https://www.mdpi.com/1424-8220/18/3/868)
- UAV + very-high-resolution satellite for wheat rust detection: [Scientific Reports 2023](https://www.nature.com/articles/s41598-023-43770-y)
- Sentinel-2 time series to *predict* stripe rust occurrence: [Agriculture 2021, 11(11), 1079](https://ideas.repec.org/a/gam/jagris/v11y2021i11p1079-d670171.html)
- UAV multispectral yellow-rust monitoring: [Computers and Electronics in Agriculture](https://www.sciencedirect.com/science/article/pii/S0168169918312584)

**Data hygiene rules**
1. Download a **license/README snapshot** for every dataset on Day 1 — you will need it for the ethics/data-availability statement.
2. Immediately hash-split into train/val/test **and** record the split file in git. Reviewers ask; you must be able to reproduce.
3. **De-duplicate** near-identical images across classes and splits. Duplicate leakage is the most common reason papers like this get rejected — and it is why so many report 99% accuracy.
4. Keep one dataset **entirely unseen** for cross-domain testing. Never tune on it.

---

## 4. What to reuse vs. build yourself (and how to do it legally)

**Rule:** reuse *infrastructure*, invent the *science*.

| Reuse as-is (fast, low risk) | Build yourself (this is the paper) |
|---|---|
| Model libraries: `timm`, `torchvision`, `ultralytics` (YOLO) | The dataset harmonisation + de-duplication protocol |
| Training loops: PyTorch Lightning / HuggingFace `Trainer` | The class taxonomy you froze |
| Grad-CAM: `pytorch-grad-cam` | Severity grading method |
| Web: Streamlit / Gradio / FastAPI | Cross-domain evaluation design + results |
| Maps: Folium / Leaflet | The map + alert app UX |
| GEE snippets for Sentinel-2 indices | Domain-gap analysis and the deployment story |

**Licensing discipline (do this or you risk a legal problem, not just a plagiarism one):**
- MIT / Apache-2.0 / BSD → safe to reuse with attribution. Note the license in your README.
- GPL / AGPL → your derivative may be forced open-source. Avoid for the app layer, or comply.
- **No license file = not actually licensed for reuse.** Contact the author or reimplement.
- Datasets have *their own* licenses (often CC-BY-NC or research-only). Check before publishing images in the paper.

**Attribution you must include**
- A `CREDITS.md` / "Acknowledgements" listing every repo used with author + license
- In the paper: cite the original method papers (timm backbones, Grad-CAM, YOLO, etc.)
- Never paste code without checking it does what you claim in the paper

---

## 5. Day-by-day workflow (5 days)

Parallelise if you have teammates. Roles: **A = Data/Model**, **B = App/Deploy**, **C = Paper/Writing**. One person can do all three at reduced depth.

### Day 1 — Foundations (do not write any app code today)

**Morning (2h): freeze decisions.** Classes, severity buckets, target venue (Section 6), team roles. Write them into `DECISIONS.md` and stop debating.

**A — Data (6h)**
- Download 2+ datasets. Record license, source URL, image counts.
- Run the **de-duplication + leakage audit**: perceptual hashing (`imagehash`) to find near-duplicates; remove cross-split duplicates.
- Build a single unified folder: `data/processed/{train,val,test}/{class}/*.jpg` + `splits.csv`.
- Produce the **Dataset Card**: counts per class per split, source, license, known biases.
- Baseline sanity check: how many images per class? If any class < 200, either merge it or drop it now.

**B — Scaffold (6h)**
- Repo init, `requirements.txt`, `.gitignore`, README skeleton.
- Get a **"hello world" end-to-end path working even with a dummy model** (2 classes, random weights): upload image → get prediction → see it on a map. Prefer having the plumbing done before the real model exists.
- Set up free hosting accounts (Hugging Face Spaces / Streamlit Community Cloud / Render). Verify you can deploy a trivial app today. Deployment surprises on Day 4 are fatal.
- Set up a GitHub repo + a project board with the tasks below.

**C — Literature (4h)**
- Collect **25–35 papers** into a reference manager (Zotero). Snowball from the papers in Section 3.
- Fill a **literature matrix** in a spreadsheet: `citation | dataset | classes | method | reported accuracy | evaluated cross-dataset? | deployed? | limitation`. The empty cells in the last two columns are *your* contribution. This matrix becomes both Related Work and your novelty argument.
- Write a **one-paragraph gap statement** and pin it. Everything you do from here must serve it.

**End of Day 1 gate:** unified dataset on disk with a clean split + deployed hello-world app + gap statement. If not, cut a class and move on.

### Day 2 — Model + baseline

**A (8h)**
- Train baseline 1: pretrained EfficientNet-B0 or ResNet-50 via `timm`, simple augmentation. ~15 epochs, 224px.
- Train baseline 2: a second family (e.g. ConvNeXt-Tiny or a ViT/DeiT) for comparison. **Comparison across families is what turns a project into a paper.**
- Log everything (TensorBoard or Weights & Biases). Save `results/baseline_*.csv`.
- Save confusion matrix + per-class F1.

**B (8h)**
- Build real inference API: preprocessing (resize, normalize, optional CLAHE), model load, `/predict` endpoint returning `{class, confidence, severity, advisory}`.
- Wire the app: upload → predict → visual result. Add **Grad-CAM overlay** (cheap, big visual impact, reviewers like explainability).
- Add the **severity grader** (start with a simple heuristic: lesion-area ratio via colour segmentation → bucket). Document it as heuristic; this is defensible if you say so.
- Map tab: report pins, clustering, list of recent reports, weather/date stamp.

**C (8h)**
- Draft Introduction + Related Work from the literature matrix. Cite as you write — never write then "add citations later".
- Draft the Methods section skeleton (dataset, preprocessing, architecture, training config, metrics, evaluation protocol).

**End of Day 2 gate:** two trained baselines with numbers + app that predicts real images + Methods draft.

### Day 3 — Improvements + the scientific contribution

**A (8h)**
- Improvement round: better augmentation (RandAugment, MixUp/CutMix), class weighting / focal loss for imbalance, progressive resizing, LR schedule, maybe TTA at inference.
- **The key experiment:** evaluate on the *held-out different-domain* dataset. Report the accuracy drop. This number is gold — it is the honest finding most papers hide.
- Try one domain-generalisation fix (e.g. stronger colour/style augmentation, fine-tune on a small field subset). Report whether it helped.
- Ablation table: backbone × augmentation × resolution. Keep it small and clean.
- Export model → ONNX + a quantised/Optimised version for mobile. Measure inference latency on CPU.

**B (8h)**
- Batch upload for drone tiles: process a folder/zip of tiles, produce a **disease heatmap grid** overlay — this is your "mapping" deliverable.
- Alert rules: if a disease class fires above confidence threshold in N reports within a radius → generate advisory banner. Keep it rule-based and explainable.
- Responsive polish: test on an actual phone browser. Fix the upload flow.
- Add a **feedback loop**: users can flag wrong predictions → stored to a CSV. Costs 30 min, gives you a great "future work / human-in-the-loop" paragraph.

**C (8h)**
- Draft Results with placeholder tables/figures, then fill from Day 3 logs on Day 4.
- Draft Discussion + Limitations. Be explicit: lab-like training data, no pathogen-level validation, no yield-loss modelling, rule-based advisory, class imbalance, single-crop scope.
- Create the figure plan: Figure 1 architecture diagram (**draw it yourself** — do not trace another paper's), Fig 2 dataset samples, Fig 3 confusion matrix, Fig 4 Grad-CAM, Fig 5 app screenshots, Fig 6 map view, Fig 7 domain-gap bar chart.

**End of Day 3 gate:** final model chosen with an honest cross-domain number + app fully functional + all paper sections drafted.

### Day 4 — Freeze, evaluate, deploy, polish

**Morning — FREEZE THE MODEL.** No more training. Scope creep here is the #1 schedule killer.

**A (5h)**
- Re-run the frozen model → final, clean, reproducible evaluation. One command → one results file. Save the random seed.
- Produce final figures at publication DPI (300) in vector where possible.
- Write `REPRODUCE.md`: exact commands + expected outputs.

**B (5h)**
- Deploy the real model to production. Verify the public URL works on mobile data, not just your laptop.
- Add a landing section explaining what it does + a demo video (60–90s screen recording) — reviewers and the demo both benefit.
- Load-test lightly; handle the "no GPS in EXIF" case gracefully.

**C (6h)**
- Assemble the full manuscript in the target template. Abstract → Conclusion pass.
- Write the **Data Availability** + **Code Availability** statements with real repo links and licences.
- Write the **Ethics / Acknowledgements** statement, naming every reused repo and dataset.

**End of Day 4 gate:** deployed public app + complete manuscript in template + all figures final.

### Day 5 — Originality screening, preprint, submission

1. **Similarity check (mandatory, 2h).** Run the manuscript through Turnitin/iThenticate or your institution's checker. Target **< 10% overall similarity** and **0% from any single source**. Rewrite anything flagged — paraphrase *and* restructure, and cite properly. This is also your defence: keep the report.
2. **AI-assistance declaration.** Most publishers now require disclosure if you used AI tools for writing. Declare it per the venue's policy — undisclosed AI text is grounds for desk rejection.
3. **Verify every citation is real.** Open each one. Do not cite papers you have not read the abstract of — fabricated citations are an instant credibility kill and a plagiarism-adjacent integrity issue.
4. **Post an arXiv preprint** (cs.CV + optionally q-bio/agriculture). Gets you a DOI and a priority timestamp today, while the journal review runs. Check the venue's preprint policy first — most allow it, some don't.
5. **Submit to the journal/conference.** Submission is your Day-5 deliverable; acceptance is a later, separate event.
6. Prepare the **response-to-reviewers fund**: keep the repo, splits.csv, and one-command reproduction working. You'll need them in 3 weeks.

---

## 6. Venue strategy

Pick based on what you actually need (deadline vs. prestige vs. speed).

| Path | Speed | Cost | Notes |
|---|---|---|---|
| **arXiv preprint now + journal later** | Immediate | Free | Best 5-day move. Citable immediately. Do this regardless. |
| **MDPI *Agriculture* / *Agronomy* / *Plants*** | Fastest realistic journal | APC — *Agriculture* is around CHF 2,600; verify current APC on the journal site | Rationale: fast decisions, receptive to applied AI+agriculture. Check current APC and whether your institution has an MDPI agreement (many do — can be free). |
| **Other MDPI titles** | e.g. *IJ Plant Biology* region of ~17.5 days first decision | Varies | Verify the journal is indexed in Scopus/WoS before submitting. |
| **IEEE/Springer conference** | Review 4–12 weeks, submission deadline may fit | Lower/no APC | Good if a relevant conference has an open CFP. Check the actual CFP page — results from generic searches are unreliable. |
| **Student/regional workshop** | Often fastest acceptance | Free/cheap | Lower prestige, but real, and fine as a first paper. |
| **Predatory journals ("accept in 48 hours")** | Days | Money | **Avoid.** They damage your record permanently and are often not indexed. If a journal guarantees acceptance before review, walk away. |

**Verify before submitting:** current APC, indexing, scope, and whether a preprint is permitted. Journal APCs and timelines change — always check the official journal page on submission day.

**What makes this publishable rather than "yet another 99% CNN":**
1. **Multi-source, leakage-audited dataset** with a published split file.
2. **Honest cross-domain evaluation** — you measure and report the lab→field drop.
3. **A domain-generalisation attempt** with an ablation showing what helped.
4. **A deployed system** with real screenshots and latency numbers — most papers stop at a notebook.
5. **Explainability** (Grad-CAM) + a documented advisory rule engine.
6. Clearly stated limitations instead of inflated claims.

Items 1–3 are your novelty. Items 4–5 are your demo value. Item 6 protects you in review.

---

## 7. Anti-plagiarism protocol (non-negotiable)

**Safe**
- Reusing open-source *code* with attribution and licence compliance
- Citing other work and building on it
- Writing your own description of someone else's method and citing them
- Using public datasets per their licence

**Unsafe — will get you flagged**
- Copying any sentence from a paper, blog, or GitHub README without quotes + citation
- Reusing figures/tables/screenshots from other papers (even "just to show the disease") — make your own or get written permission
- "Paraphrasing" by swapping synonyms while keeping sentence structure
- Submitting text generated by an undisclosed AI tool where the venue forbids or requires disclosure
- Self-plagiarism: reusing your own earlier published text without citing it
- Citing papers you have not read (fabricated citations are the fastest way to be rejected)

**Practical workflow**
1. Write every sentence yourself, in your own structure. Use source papers for *facts*, never for phrasing.
2. Never copy-paste between your notes and the manuscript. Paste into a notes file, then rewrite from scratch.
3. Keep a "source → my sentence" trace next to every claim you make from literature.
4. Run the similarity check on Day 5 and *keep the report*.
5. Draw all figures from your own data. Architecture diagrams: draw in draw.io/Excalidraw yourself.
6. Disclose AI assistance per the venue policy. When in doubt, disclose — it costs a line, non-disclosure can cost the paper.

---

## 8. Risk register

| Risk | Mitigation |
|---|---|
| Dataset too small / imbalanced | Freeze classes Day 1 after counting; drop classes under ~200 images; class weighting + focal loss |
| Duplicate images inflate accuracy | Perceptual-hash de-duplication on Day 1 — non-negotiable |
| Deployment fails on Day 4 | Verify a dummy deploy on Day 1; keep Streamlit Cloud as fallback |
| Model won't converge by Day 3 | Fall back to EfficientNet-B0 @224px + heavy augmentation; it nearly always works |
| Training hardware too slow | Use Google Colab free GPU / Kaggle notebooks (30 GPU-h/week free) — check current limits |
| Satellite/GEE work overruns | It is *bonus*. Do Sentinel-2 indices only if Day 3 finishes early. Cut without guilt. |
| Paper exceeds page limit | Check the template early; put architecture/ablation detail in a supplement |
| Low similarity but still "sounds copied" | Restructure paragraphs, change sentence order, lead with your own finding |
| Reviewer asks for field-validated data | Already covered in Limitations — say it plainly and propose it as future work |

---

## 9. Repository layout to build

```
wheat-disease-ai/
├── PLAN.md              # this file
├── DECISIONS.md         # frozen scope: classes, buckets, venue
├── CREDITS.md           # every reused repo/dataset + licence
├── REPRODUCE.md         # exact commands to reproduce every number
├── data/
│   ├── raw/             # untouched downloads (gitignored)
│   ├── processed/       # unified class folders (gitignored)
│   └── splits.csv       # THE split file — commit this
├── notebooks/           # exploration + figures
├── src/
│   ├── data/            # dedup.py, build_dataset.py, dataset.py
│   ├── models/          # backbones, train.py, evaluate.py
│   ├── explain/         # gradcam.py
│   ├── severity/        # lesion-area heuristic
│   ├── geo/             # sentinel2 indices (bonus)
│   └── advisory/        # class → action rules
├── app/
│   ├── app.py           # Streamlit UI (predict / map / batch)
│   ├── api.py           # FastAPI inference endpoint
│   └── static/          # screenshots for the paper
├── models/              # exported weights + ONNX
├── results/             # metrics, confusion matrices, logs
├── paper/
│   ├── manuscript.tex   # or .docx
│   ├── figures/         # 300 DPI, original
│   └── refs.bib
└── requirements.txt
```

---

## 10. Master checklist

**Day 1** — scope frozen; 2+ datasets downloaded with licences; de-duplicated splits; Dataset Card; hello-world app deployed; 25+ papers in Zotero; literature matrix; gap statement.
**Day 2** — 2 baselines trained with numbers; real inference API; Grad-CAM; severity grader; map tab; Intro + Related Work + Methods drafted.
**Day 3** — improvement ablation; **cross-domain result**; ONNX export + latency; drone batch heatmap; alert rules; mobile polish; all paper sections drafted.
**Day 4** — model frozen; final reproducible evaluation; 300 DPI figures; public deploy verified on mobile; manuscript assembled; data/code/ethics statements.
**Day 5** — similarity check < 10%; AI disclosure; all citations verified as real; arXiv preprint posted; journal submission sent.

---

## 11. Your immediate next actions (next 60 minutes)

1. Decide: **5 classes (Set A) or 7 (Set B)?** If unsure → Set A.
2. Decide: **do you have teammates, and a GPU?** That changes the split of labour.
3. Pick the target venue and check its official page for current APC + deadline.
4. Create the GitHub repo and the folder skeleton above.
5. Download the Kaggle "Wheat Plant Diseases" dataset and count images per class. That single number determines whether your plan is realistic.
