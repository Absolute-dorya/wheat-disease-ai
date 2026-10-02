"""
Train a wheat-disease classifier via transfer learning (timm).

Expects:  data/processed/{train,val,test}/<class>/<image>
Produces: results/<run>/best.pt, metrics.json, confusion_matrix.png, log.csv

Example:
    python src/train.py --data data/processed --backbone efficientnet_b0 --epochs 15
    python src/train.py --data data/processed --backbone convnext_tiny --epochs 15
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int, strong: bool):
    """`strong=True` adds colour/style jitter -> the main tool against the
    lab-to-field domain gap. Ablate this in the paper."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if strong:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.15),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    losses, preds, labels = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            losses.append(criterion(out, y).item() * x.size(0))
        preds.extend(out.argmax(1).cpu().tolist())
        labels.extend(y.cpu().tolist())
    n = max(len(labels), 1)
    return {
        "loss": (sum(losses) / n) if losses else None,
        "preds": preds,
        "labels": labels,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--strong-aug", action="store_true",
                    help="Enable colour/style jitter augmentation")
    ap.add_argument("--class-weights", action="store_true",
                    help="Weight loss inversely to class frequency")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    import timm  # imported here so --help works without torch installed

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] device={device}  backbone={args.backbone}")

    root = Path(args.data)
    train_tf, eval_tf = build_transforms(args.img_size, args.strong_aug)

    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
    classes = train_ds.classes
    print(f"[i] classes ({len(classes)}): {classes}")
    print(f"[i] train={len(train_ds)}  val={len(val_ds)}")

    # Sanity gate: warn loudly on thin classes (they will drag macro-F1 down).
    counts = pd.Series([s[1] for s in train_ds.samples]).value_counts().sort_index()
    for idx, c in enumerate(classes):
        n = int(counts.get(idx, 0))
        if n < 100:
            print(f"[!] class '{c}' has only {n} training images -- consider merging it")

    pin = device == "cuda"
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=pin)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=pin)

    model = timm.create_model(args.backbone, pretrained=True, num_classes=len(classes))
    model.to(device)

    weight = None
    if args.class_weights:
        cw = (counts.sum() / (len(classes) * counts.clip(lower=1))).values.astype("float32")
        weight = torch.tensor(cw, device=device)
        print(f"[i] class weights: {dict(zip(classes, weight.cpu().numpy().round(3)))}")

    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=pin)

    run = args.run_name or f"{args.backbone}_{args.img_size}_{'sa' if args.strong_aug else 'ba'}_seed{args.seed}"
    out_dir = Path(args.out) / run
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1, history, t0 = -1.0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=pin):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        sched.step()

        tr = running / max(seen, 1)
        va = evaluate(model, val_ld, device, criterion)
        f1 = f1_score(va["labels"], va["preds"], average="macro", zero_division=0)
        acc = float(np.mean(np.array(va["preds"]) == np.array(va["labels"])))

        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va["loss"],
                        "val_acc": acc, "val_macro_f1": f1})
        print(f"[{epoch:>3}/{args.epochs}] train_loss={tr:.4f} "
              f"val_loss={va['loss']:.4f} val_acc={acc:.4f} val_macro_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "backbone": args.backbone, "img_size": args.img_size,
                        "strong_aug": args.strong_aug, "epoch": epoch,
                        "val_macro_f1": f1},
                       out_dir / "best.pt")

    pd.DataFrame(history).to_csv(out_dir / "log.csv", index=False)

    # ---- Final report on the best checkpoint -------------------------------
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    summary = {"backbone": args.backbone, "img_size": args.img_size,
               "strong_aug": args.strong_aug, "class_weights": args.class_weights,
               "seed": args.seed, "epochs": args.epochs,
               "best_val_macro_f1": best_f1, "classes": classes,
               "minutes": round((time.time() - t0) / 60, 2)}

    for split in ("val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        ds = datasets.ImageFolder(split_dir, transform=eval_tf)
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
        res = evaluate(model, ld, device, criterion)
        # Align label order if the test folder has a different class order
        labels_here = ds.classes
        rep = classification_report(res["labels"], res["preds"],
                                    labels=list(range(len(labels_here))),
                                    target_names=labels_here, digits=4,
                                    zero_division=0)
        summary[f"{split}_acc"] = float(np.mean(
            np.array(res["preds"]) == np.array(res["labels"])))
        summary[f"{split}_macro_f1"] = f1_score(res["labels"], res["preds"],
                                                average="macro", zero_division=0)
        (out_dir / f"report_{split}.txt").write_text(rep)
        print(f"\n===== {split.upper()} =====\n{rep}")

        cm = confusion_matrix(res["labels"], res["preds"],
                              labels=list(range(len(labels_here))))
        pd.DataFrame(cm, index=labels_here, columns=labels_here).to_csv(
            out_dir / f"confusion_{split}.csv")

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] artifacts in {out_dir}")


if __name__ == "__main__":
    main()
