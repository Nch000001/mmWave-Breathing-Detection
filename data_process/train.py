import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Dataset
# ----------------------------
class SegDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()  # (N, T)
        self.y = torch.from_numpy(y).long()   # (N,)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        x = self.X[i].unsqueeze(0)  # (1, T)
        return x, self.y[i]

# ----------------------------
# Model: small 1D CNN
# ----------------------------
class BreathCNN(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T/2

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T/4

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B,64,1)
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        z = self.net(x).squeeze(-1)  # (B,64)
        return self.fc(z)

# ----------------------------
# Helpers
# ----------------------------
def split_by_file(meta, val_ratio=0.2, seed=0):
    """
    Support both:
    1) old meta format: tuple/list like (file, src, center_idx, label)
    2) new meta format: dict like {"file":..., "kind":..., ...}
    """
    files_list = []

    for m in meta:
        if isinstance(m, dict):
            files_list.append(m["file"])
        elif isinstance(m, (list, tuple)):
            files_list.append(m[0])
        else:
            raise TypeError(f"Unsupported meta item type: {type(m)}")

    files = np.array(files_list, dtype=object)
    uniq = np.unique(files)

    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n_val = max(1, int(round(len(uniq) * val_ratio)))
    val_files = set(uniq[:n_val].tolist())

    is_val = np.array([f in val_files for f in files], dtype=bool)
    return is_val, sorted(list(val_files))

def metrics_from_logits(logits, y_true):
    pred = logits.argmax(axis=1)
    y = y_true
    acc = float((pred == y).mean())
    # precision/recall for class 1
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return acc, float(prec), float(rec), float(f1), tp, fp, fn

# ----------------------------
# Train
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_segments.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="breathclf_best.pt")
    args = ap.parse_args()

    pack = torch.load(args.data, map_location="cpu")
    X = pack["X"]
    y = pack["y"]
    meta = pack.get("meta", None)

    # normalize per-segment (robust enough, prevents amplitude bias)
    X = X - np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - np.median(X, axis=1, keepdims=True)), axis=1, keepdims=True) + 1e-6
    X = X / (1.4826 * mad)

    # split
    if meta is not None:
        is_val, val_files = split_by_file(meta, val_ratio=args.val_ratio, seed=args.seed)
        print("[INFO] file-level val files:", val_files)
    else:
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(X))
        rng.shuffle(idx)
        n_val = int(round(len(X) * args.val_ratio))
        is_val = np.zeros(len(X), dtype=bool)
        is_val[idx[:n_val]] = True
        print("[WARN] meta not found -> random split")

    Xtr, ytr = X[~is_val], y[~is_val]
    Xva, yva = X[is_val], y[is_val]

    tr_ds = SegDataset(Xtr, ytr)
    va_ds = SegDataset(Xva, yva)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)
    print(f"[INFO] train={len(tr_ds)} val={len(va_ds)}  pos_train={(ytr==1).sum()} neg_train={(ytr==0).sum()}")

    model = BreathCNN(T=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_f1 = -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        # val
        model.eval()
        all_logits = []
        all_y = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                logits = model(xb).cpu().numpy()
                all_logits.append(logits)
                all_y.append(yb.numpy())
        all_logits = np.concatenate(all_logits, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        acc, prec, rec, f1, tp, fp, fn = metrics_from_logits(all_logits, all_y)

        print(f"Epoch {ep:03d} | train_loss {np.mean(tr_losses):.4f} | val acc {acc:.3f} f1 {f1:.3f} (P {prec:.3f} R {rec:.3f}) tp {tp} fp {fp} fn {fn}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "T": X.shape[1]}, args.out)
            print(f"  [SAVE] best -> {args.out} (f1={best_f1:.3f})")

    print("[DONE] best_f1 =", best_f1)

if __name__ == "__main__":
    main()