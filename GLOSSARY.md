# Plain-English Glossary

Every piece of jargon from [PLAN.md](PLAN.md), explained like you've never touched machine learning. Keep this open while you work.

---

## The AI part

**Model / classifier**
The program that looks at a photo and says "this is stripe rust". It's just a very complicated pattern-matcher. You don't write rules — you show it thousands of photos with the answer attached, and it works out the pattern itself.

**Training**
That learning process. You feed examples, it adjusts itself, repeat. Nothing more mystical than that.

**CNN (convolutional neural network)**
The standard type of model for photos. It scans the image in small patches and builds up understanding in layers: edges → spots → "this has rust-coloured patches".

**Backbone**
The reusable photo-understanding part of the model. Think of it as a generic "image reader" that Google or Meta already trained on millions of pictures. You snap your own small wheat-disease classifier onto the end of it — so you're **not** starting from zero.

**Transfer learning**
Using someone else's already-trained backbone instead of training from scratch. This is why your project is doable in 5 days instead of 5 months.

**timm**
A free library that hands you ~1,000 ready-made backbones. You just pick one by name, e.g. `efficientnet_b0`.

**Epoch**
One full pass through all your training photos. "15 epochs" = the model looked at your entire dataset 15 times.

**Augmentation**
You never have enough photos, so you manufacture more for free: flip them, rotate slightly, shift the brightness. The model gets more robust because it has seen more variety.

**Grad-CAM**
A heatmap overlaid on the photo showing **which part** the model looked at when deciding. It stops the model being a black box. Cheap to add, looks good in a paper.

**Overfitting**
When the model memorises your training photos instead of learning the general pattern. It scores great on data it has seen and badly on anything new. The classic way students accidentally get fake 99% results.

---

## The data part

**Leakage — the #1 disaster**
If the same (or nearly the same) photo ends up in both your "practice" pile and your "final exam" pile, the model has already seen the answer. Your accuracy looks amazing and is completely fake. This is the most common reason papers like this get rejected.

**Duplicate / de-duplication (dedup)**
Finding those near-identical photos and removing them.

**Perceptual hash**
A photo fingerprint. Two images with similar fingerprints are essentially the same picture, even if one is resized or compressed. That's how `src/dedup.py` finds duplicates.

**Split (train / val / test)**
Splitting your photos into three piles:
- **train** — what it learns from
- **val** — used while learning, to check progress
- **test** — the final exam. Looked at ONCE, at the end. Never tune anything on it.

**Dataset**
Just a collection of photos, organised in folders by class.

**Domain gap / cross-domain**
Photos taken in a lab (plain background, even light, one leaf) look nothing like photos taken in a real field (mud, shadows, other plants, wind, angle). A model trained on lab photos often fails badly in a field. **Measuring that drop is the single most valuable thing in your project** — every paper that reports a shiny 99% is usually hiding it.

**UAV**
Drone. (Unmanned aerial vehicle.)

**Sentinel-2**
Free satellite images from the European Space Agency. You don't need to own a satellite.

**NDVI**
A standard arithmetic formula applied to satellite images that tells you how green/healthy the vegetation is. High = lush, low = stressed or diseased.

**Multispectral / hyperspectral**
Cameras that see wavelengths our eyes can't (near-infrared etc.). Diseased plants change there before they look visibly sick — that's why researchers use these instead of ordinary phone cameras.

---

## The scoring part

**Accuracy**
Percentage of photos it got right. Sounds good, but misleading when your classes are unequal — if 90% of your photos are healthy, a lazy model that always says "healthy" scores 90%.

**Precision and recall**
- **Precision** — when it says "rust", how often is it actually rust?
- **Recall** — of all the genuinely rust-infected plants, how many did it catch?

Both matter: a model that misses half the rust cases is useless even at 95% accuracy.

**Confusion matrix**
A grid showing what it got right, and crucially **what it confused things with** when wrong. Rust↔Septoria confusion is the usual weakness in wheat models. This table is one of your paper's figures.

**Macro-F1**
One number that averages performance across all classes, treating a rare disease as just as important as a common one. **Better than accuracy — report this one.**

**Ablation**
Switching one thing off at a time to prove it actually helped. If you claim "strong augmentation helped", you must show results with it off and on. Without ablations, reviewers say "you have no idea why your model works".

**Baseline**
The simple starting model you compare everything against. You need at least two, or you have nothing to compare your improvements to.

**Latency**
How long the model takes to answer, usually in milliseconds. Matters because farmers are on phones, not gaming PCs.

---

## The app / deployment part

**Deploy**
Put it on the internet so a farmer can open a link on their phone and use it. A script running on your laptop is **not** deployed.

**Streamlit**
A Python library that turns a plain script into a working website with almost no web programming. That's what `app/app.py` uses.

**Folium**
The library that draws the interactive map with pins on it.

**API**
The middle layer the app talks to when it wants the model's answer. Keeps things separate: the website asks, the model answers.

**Batch processing**
Uploading many photos at once (e.g. all the tiles from one drone flight) and getting all results back.

**ONNX**
A standard file format for a finished model, so it can run on phones and other systems instead of only on your laptop.

**PWA (progressive web app)**
A website that behaves like an app on a phone. Gets you "works on mobile" without building a real Android/iOS app — which you do not have time for.

---

## The paper part

**Preprint (arXiv)**
You post your paper publicly **before** peer review. You get a public link and a timestamp **on Day 5**, while the journal review runs for weeks. Most journals allow this — check yours first.

**Peer review**
Other researchers read your paper, criticise it, and recommend accept or reject. Slow: weeks to months, on the publisher's clock, not yours.

**First decision**
The journal's first verdict: accept / minor revisions / major revisions / reject. "Major revisions" is a normal, good outcome — not a failure.

**APC (article processing charge)**
The fee to publish open access. Can be thousands of USD/CHF per paper. **Check whether your institution has a deal that covers it** — many universities do, which makes it free for you.

**Predatory journal**
A pay-to-publish operation that "accepts" without real review, often within days. They damage your record permanently and often aren't indexed anywhere. If a journal guarantees acceptance before review, walk away.

**Similarity check (Turnitin / iThenticate / "plagiarism check")**
Software that flags text matching existing sources. Run it before you submit and **keep the report** — it's your evidence you did the work.

**Zotero**
Free software to store, organise and auto-format the papers you cite. Do not manage 35 references in a text file.

**Data availability statement**
A required paragraph saying where your data came from and how others can get it.

**Code availability statement**
Same idea for your code: a public repo link, a licence, and the exact version you used.

---

## Quick rule of thumb

If a sentence in your paper makes a claim about another person's work, it needs a **citation**. If it describes something **you** did or found, it must be written in **your own words** — never pasted. That one rule covers 90% of plagiarism risk.
