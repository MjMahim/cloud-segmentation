"""Train a U-Net (pre-trained ResNet34 encoder) for cloud segmentation on 38-Cloud.

Examples:
    python train.py --bands rgbn --epochs 20
    python train.py --bands rgb  --epochs 20
    python train.py --bands rgbn --epochs 1 --limit-batches 30   # quick speed test
"""
import argparse
import csv
import time
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from data import CachedCloudDataset, nonempty_train_patches, split_by_scene


def confusion(logits, mask):
    """True/false positives and false negatives at probability threshold 0.5."""
    pred = logits > 0  # sigmoid(x) > 0.5  <=>  x > 0
    gt = mask > 0.5
    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()
    tn = (~pred & ~gt).sum().item()
    return tp, fp, fn, tn


def scores(tp, fp, fn, tn):
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"iou": iou, "precision": precision, "recall": recall, "accuracy": accuracy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", choices=["rgb", "rgbn"], default="rgbn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-batches", type=int, default=0, help="stop each epoch early (for speed tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt (or best.pt)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(__file__).parent / "runs" / f"unet_{args.encoder}_{args.bands}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_names, val_names, val_scenes = split_by_scene(nonempty_train_patches())
    print(f"train patches {len(train_names)}, val patches {len(val_names)} (val scenes: {len(val_scenes)})")

    train_dl = DataLoader(CachedCloudDataset(train_names, args.bands, augment=True), batch_size=args.batch,
                          shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(CachedCloudDataset(val_names, args.bands), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    in_ch = 3 if args.bands == "rgb" else 4
    # With in_channels=4, smp adapts the ImageNet first layer to 4 inputs (extra channel initialised from the others)
    model = smp.Unet(args.encoder, encoder_weights="imagenet", in_channels=in_ch, classes=1).to(device)

    dice = smp.losses.DiceLoss(mode="binary")
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    history, best_iou, start = [], -1.0, 1
    if args.resume:
        last, best = run_dir / "last.pt", run_dir / "best.pt"
        if last.exists():  # full training state saved after every epoch
            ck = torch.load(last, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            history, best_iou, start = ck["history"], ck["best_iou"], ck["epoch"] + 1
        elif best.exists():  # older run without last.pt: continue from the best epoch's weights
            ck = torch.load(best, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            best_iou, start = ck["val"]["iou"], ck["epoch"] + 1
            with open(run_dir / "history.csv", newline="") as f:
                history = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)][:ck["epoch"]]
            for _ in range(ck["epoch"]):
                sched.step()  # put the learning-rate schedule where it was
        print(f"resuming at epoch {start} (best val IoU so far {best_iou:.4f})")

    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, n = 0.0, 0
        for b, (x, y) in enumerate(train_dl):
            if args.limit_batches and b >= args.limit_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(x)
                loss = bce(logits, y) + dice(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item() * len(x)
            n += len(x)
        sched.step()
        train_time = time.time() - t0

        model.eval()
        totals = [0, 0, 0, 0]
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            for b, (x, y) in enumerate(val_dl):
                if args.limit_batches and b >= args.limit_batches:
                    break
                logits = model(x.to(device, non_blocking=True)).float()
                totals = [t + c for t, c in zip(totals, confusion(logits.cpu(), y))]
        val = scores(*totals)

        row = {"epoch": epoch, "train_loss": train_loss / max(n, 1), **{f"val_{k}": v for k, v in val.items()},
               "seconds": time.time() - t0}
        history.append(row)
        print(f"epoch {epoch:2d}  loss {row['train_loss']:.4f}  val IoU {val['iou']:.4f}  "
              f"P {val['precision']:.3f}  R {val['recall']:.3f}  ({train_time:.0f}s train, {row['seconds']:.0f}s total)",
              flush=True)

        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val": val}, run_dir / "best.pt")

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "history": history, "best_iou": best_iou,
                    "args": vars(args)}, run_dir / "last.pt")

        with open(run_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)

    print(f"best val IoU {best_iou:.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
