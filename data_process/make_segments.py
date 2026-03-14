import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from label_gated import build_roi_waveform


def parse_kind(name: str) -> str:
    s = name.lower()
    if "slow" in s:
        return "slow"
    if "normal" in s:
        return "normal"
    if "fast" in s:
        return "fast"
    if "mix" in s:
        return "mix"
    if "talk" in s:
        return "talk"
    if "move" in s:
        return "move"
    if "bg" in s or "empty" in s or "background" in s:
        return "bg"
    return "unknown"


def read_file_list(path: Path):
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def cut_segment(sig: np.ndarray, center: int, half: int):
    if center - half < 0 or center + half >= len(sig):
        return None
    return sig[center - half:center + half].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_dir", required=True, help="positive source dir: slow/normal/fast/mix")
    ap.add_argument("--bg_dir", required=True, help="negative source dir: talk/move/empty/bg")
    ap.add_argument("--events_dir", required=True, help="dir containing *_events.csv from teacher")
    ap.add_argument("--out", default="dataset_segments_v2.pt")

    ap.add_argument("--file_list", default=None, help="optional txt file list for h5_dir")
    ap.add_argument("--use_mix_as_pos", action="store_true", help="include mix as positive source (default: false)")

    ap.add_argument("--dt", type=float, default=0.035)
    ap.add_argument("--f_lo", type=float, default=0.10)
    ap.add_argument("--f_hi", type=float, default=0.90)
    ap.add_argument("--roi_topk", type=float, default=0.05)

    ap.add_argument("--seg_sec", type=float, default=3.0)
    ap.add_argument("--neg_stride_sec", type=float, default=1.5, help="stride for bg negative sliding windows")
    ap.add_argument("--max_neg_per_file", type=int, default=80, help="limit negatives per bg file")
    args = ap.parse_args()

    fs = 1.0 / args.dt
    seg_len = int(round(args.seg_sec * fs))
    if seg_len % 2 == 1:
        seg_len += 1
    half = seg_len // 2
    neg_stride = int(round(args.neg_stride_sec * fs))
    neg_stride = max(1, neg_stride)

    h5_dir = Path(args.h5_dir)
    bg_dir = Path(args.bg_dir)
    events_dir = Path(args.events_dir)

    # -------- positive file list --------
    if args.file_list:
        pos_names = read_file_list(Path(args.file_list))
        pos_files = [h5_dir / f"{n}.h5" for n in pos_names]
    else:
        pos_files = sorted(h5_dir.glob("*.h5"))

    # keep only stable breathing as positive by default
    filtered_pos_files = []
    for p in pos_files:
        kind = parse_kind(p.stem)
        if kind in ("slow", "normal", "fast"):
            filtered_pos_files.append(p)
        elif kind == "mix" and args.use_mix_as_pos:
            filtered_pos_files.append(p)

    bg_files = sorted(bg_dir.glob("*.h5"))

    X = []
    y = []
    meta = []

    # =========================
    # Positive samples
    # =========================
    print("[INFO] building positive samples...")
    pos_count = 0
    for h5_path in filtered_pos_files:
        name = h5_path.stem
        kind = parse_kind(name)

        ev_path = events_dir / f"{name}_events.csv"
        if not h5_path.exists():
            print(f"[SKIP] missing h5: {h5_path}")
            continue
        if not ev_path.exists():
            print(f"[SKIP] missing events: {ev_path}")
            continue

        try:
            s_raw, s_bp, src, N, meta_info = build_roi_waveform(
                h5_path,
                ds_name="DS1",
                label_name="LABEL",
                fs=fs,
                f_lo=args.f_lo,
                f_hi=args.f_hi,
                roi_topk=args.roi_topk,
                drop_head_sec=1.5,
                use_sec=None,
            )
        except Exception as e:
            print(f"[FAIL] build waveform pos {name}: {e}")
            continue

        df = pd.read_csv(ev_path)
        if "event_idx" not in df.columns:
            print(f"[SKIP] no event_idx: {ev_path}")
            continue

        events = df["event_idx"].to_numpy(dtype=int)
        kept_here = 0
        for e in events:
            seg = cut_segment(s_bp, int(e), half)
            if seg is None:
                continue
            X.append(seg)
            y.append(1)
            meta.append({
                "file": name,
                "kind": kind,
                "src": src,
                "label": 1,
                "center_idx": int(e),
            })
            kept_here += 1

        pos_count += kept_here
        print(f"[OK] POS {name}: +{kept_here}")

    # =========================
    # Negative samples from bg
    # =========================
    print("\n[INFO] building negative samples from bg...")
    neg_count = 0
    rng = np.random.default_rng(0)

    for h5_path in bg_files:
        name = h5_path.stem
        kind = parse_kind(name)

        if not h5_path.exists():
            continue

        try:
            s_raw, s_bp, src, N, meta_info = build_roi_waveform(
                h5_path,
                ds_name="DS1",
                label_name="LABEL",
                fs=fs,
                f_lo=args.f_lo,
                f_hi=args.f_hi,
                roi_topk=args.roi_topk,
                drop_head_sec=1.5,
                use_sec=None,
            )
        except Exception as e:
            print(f"[FAIL] build waveform bg {name}: {e}")
            continue

        centers = np.arange(half, N - half, neg_stride, dtype=int)
        if len(centers) == 0:
            print(f"[SKIP] BG too short: {name}")
            continue

        # subsample if too many
        if len(centers) > args.max_neg_per_file:
            centers = rng.choice(centers, size=args.max_neg_per_file, replace=False)
            centers = np.sort(centers)

        kept_here = 0
        for c in centers:
            seg = cut_segment(s_bp, int(c), half)
            if seg is None:
                continue
            X.append(seg)
            y.append(0)
            meta.append({
                "file": name,
                "kind": kind,
                "src": src,
                "label": 0,
                "center_idx": int(c),
            })
            kept_here += 1

        neg_count += kept_here
        print(f"[OK] NEG {name}: +{kept_here}")

    if len(X) == 0:
        raise RuntimeError("No segments collected. Check h5/events/bg paths.")

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    meta_df = pd.DataFrame(meta)

    out = Path(args.out)
    torch.save({
        "X": X,
        "y": y,
        "meta": meta_df.to_dict(orient="records"),
        "fs": fs,
        "seg_len": seg_len,
    }, out)

    print(f"saved: {out}")


if __name__ == "__main__":
    main()