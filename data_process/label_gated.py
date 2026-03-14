import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import h5py

from scipy.signal import butter, sosfiltfilt, find_peaks


# ----------------------------
# Utils
# ----------------------------
def design_bandpass(fs, f_lo=0.10, f_hi=0.90, order=4):
    nyq = 0.5 * fs
    lo = max(1e-6, f_lo / nyq)
    hi = min(0.999999, f_hi / nyq)
    return butter(order, [lo, hi], btype="bandpass", output="sos")


def robust_zscore(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return (x - med) / (1.4826 * mad)


def rfft_bandpower(px_t, fs, f_lo, f_hi):
    T = px_t.shape[1]
    X = np.fft.rfft(px_t, axis=1)
    Pxx = (np.abs(X) ** 2) / max(1, T)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    idx = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
    if idx.size == 0:
        return np.zeros((px_t.shape[0],), dtype=np.float64)
    return Pxx[:, idx].sum(axis=1)


def estimate_f0_fft(seg, fs, f_lo=0.10, f_hi=0.90):

    seg = np.asarray(seg, dtype=np.float64)
    seg = seg - np.mean(seg)
    n = len(seg)

    X = np.fft.rfft(seg)
    P = (np.abs(X) ** 2) / max(1, n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    band = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
    if band.size < 3:
        return np.nan, 0.0

    Pb = P[band]
    fb = freqs[band]

    i1 = int(np.argmax(Pb))
    f1 = float(fb[i1])
    p1 = float(Pb[i1])

    def power_at(f):
        j = int(np.argmin(np.abs(fb - f)))
        return float(Pb[j]), float(fb[j])

    f_half = 0.5 * f1
    if f_half >= f_lo:
        p_half, f_half_bin = power_at(f_half)
        if p_half >= 0.30 * p1:
            med = float(np.median(Pb) + 1e-12)
            return float(f_half_bin), float(p_half / med)

    med = float(np.median(Pb) + 1e-12)
    return float(f1), float(p1 / med)


def merge_close_events(ev_idx, fs, T0, min_frac=0.60):
    if len(ev_idx) <= 1 or not np.isfinite(T0):
        return np.asarray(ev_idx, dtype=int)
    min_d = int(round(min_frac * T0 * fs))
    ev_idx = list(sorted(map(int, ev_idx)))
    kept = [ev_idx[0]]
    for i in range(1, len(ev_idx)):
        if ev_idx[i] - kept[-1] < min_d:
            continue
        kept.append(ev_idx[i])
    return np.array(kept, dtype=int)


def insert_missing_events(ev_idx, fs, T0, max_frac=1.60):
    if len(ev_idx) <= 1 or not np.isfinite(T0):
        return np.asarray(ev_idx, dtype=int)

    max_d = max(1, int(round(max_frac * T0 * fs)))
    target_d = max(1, int(round(T0 * fs)))

    ev_idx = list(sorted(map(int, ev_idx)))
    out = [ev_idx[0]]
    for i in range(1, len(ev_idx)):
        prev = out[-1]
        cur = ev_idx[i]
        gap = cur - prev
        if gap > max_d:
            # conservative: insert at most 1
            out.append(prev + target_d)
        out.append(cur)
    return np.array(sorted(set(out)), dtype=int)


def crop_by_label(arr, label, fs, drop_head_sec=1.5, use_sec=None,
            gate_min_max_votes=6, gate_min_peak_ratio=3.0, gate_min_band_ratio=0.15):

    N = arr.shape[-1]
    lab = np.asarray(label).reshape(-1)
    if lab.size != N:
        raise ValueError(f"LABEL length {lab.size} != N {N}")

    idx = np.where(lab == 1)[0]
    if idx.size == 0:
        # no valid region; return original
        return arr, 0, N

    start0 = int(idx[0])
    start = int(start0 + round(drop_head_sec * fs))
    start = max(0, min(start, N - 1))

    end = N
    if use_sec is not None:
        L = int(round(float(use_sec) * fs))
        end = min(N, start + max(1, L))

    return arr[:, :, :, start:end], start, end


def build_roi_waveform(h5_path, ds_name="DS1", label_name="LABEL",
                       fs=1/0.035, f_lo=0.10, f_hi=0.90, roi_topk=0.05,
                       drop_head_sec=1.5, use_sec=None,
            gate_min_max_votes=6, gate_min_peak_ratio=3.0, gate_min_band_ratio=0.15):
    with h5py.File(h5_path, "r") as f:
        if ds_name not in f:
            raise ValueError(f"No {ds_name} in {h5_path}")
        arr = np.asarray(f[ds_name][()])

        if label_name not in f:
            raise ValueError(f"No {label_name} in {h5_path}")
        lab = np.asarray(f[label_name][()])

    if arr.ndim != 4 or arr.shape[0] < 2:
        raise ValueError(f"Unexpected DS1 shape {arr.shape}, expect (2,32,32,N)")

    # crop by label
    arr, start, end = crop_by_label(arr, lab, fs, drop_head_sec=drop_head_sec, use_sec=use_sec)

    C, H, W, N = arr.shape
    rdi = arr[0].astype(np.float32, copy=False)
    phd = arr[1].astype(np.float32, copy=False)

    P = H * W
    rdi_px = rdi.reshape(P, N)
    phd_px = phd.reshape(P, N)

    # ROI selection by bandpower on cropped segment
    bp_rdi = rfft_bandpower(rdi_px, fs, f_lo, f_hi)
    bp_phd = rfft_bandpower(phd_px, fs, f_lo, f_hi)

    def peakiness_per_px(px_t):
        # px_t: (P, N)
        # 取每個像素在頻帶內的 peakiness = max(Pxx_band)/median(Pxx_band)
        T = px_t.shape[1]
        X = np.fft.rfft(px_t - px_t.mean(axis=1, keepdims=True), axis=1)
        Pxx = (np.abs(X) ** 2) / max(1, T)
        freqs = np.fft.rfftfreq(T, d=1.0 / fs)
        band = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(band):
            return np.ones((px_t.shape[0],), dtype=np.float64)
        Pb = Pxx[:, band]
        peak = Pb.max(axis=1)
        med = np.median(Pb, axis=1) + 1e-12
        return peak / med

    pk_rdi = peakiness_per_px(rdi_px.astype(np.float64))
    pk_phd = peakiness_per_px(phd_px.astype(np.float64))

    # ROI score: bandpower * clipped peakiness
    score_rdi = bp_rdi * np.clip(pk_rdi, 1.0, 10.0)
    score_phd = bp_phd * np.clip(pk_phd, 1.0, 10.0)

    P = H * W
    k = max(1, int(P * roi_topk))

    top_rdi = np.partition(score_rdi, -k)[-k:].mean()
    top_phd = np.partition(score_phd, -k)[-k:].mean()
    use_src = "RDI" if top_rdi >= top_phd else "PHD"

    score_use = score_rdi if use_src == "RDI" else score_phd
    roi_idx = np.argpartition(score_use, -k)[-k:]

    if use_src == "RDI":
        s_raw = rdi_px[roi_idx].mean(axis=0).astype(np.float64, copy=False)
    else:
        s_raw = phd_px[roi_idx].mean(axis=0).astype(np.float64, copy=False)

    s_raw = s_raw - np.median(s_raw)
    s_norm = robust_zscore(s_raw)

    sos = design_bandpass(fs, f_lo, f_hi, order=4)
    s_bp = sosfiltfilt(sos, s_norm).astype(np.float64)

    meta = {
        "crop_start_idx": int(start),
        "crop_end_idx": int(end),
        "crop_len": int(N),
    }
    return s_raw, s_bp, use_src, N, meta



def rfft_band_ratio(x, fs, f_lo, f_hi, f_min_total=0.05, f_max_total=2.0):

    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    if n < 8:
        return 0.0
    X = np.fft.rfft(x)
    P = (np.abs(X) ** 2) / max(1, n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    band = (freqs >= f_lo) & (freqs <= f_hi)
    total = (freqs >= f_min_total) & (freqs <= f_max_total)

    band_energy = float(P[band].sum()) if np.any(band) else 0.0
    total_energy = float(P[total].sum()) if np.any(total) else 0.0
    if total_energy <= 1e-12:
        return 0.0
    return band_energy / total_energy


def breath_presence_gate(s_raw, votes, fs, f_lo, f_hi,
                        min_max_votes=6,
                        min_peak_ratio=3.0,
                        min_band_ratio=0.15):

    v = np.asarray(votes, dtype=np.float64)
    vmax = float(v.max()) if v.size else 0.0
    vmean = float(v.mean()) if v.size else 0.0
    peak_ratio = float(vmax / (vmean + 1e-9)) if v.size else 0.0

    band_ratio = float(rfft_band_ratio(s_raw, fs, f_lo, f_hi))

    # hard decision
    has_breath = (vmax >= float(min_max_votes)) and (peak_ratio >= float(min_peak_ratio)) and (band_ratio >= float(min_band_ratio))

    # soft confidence in [0,1]
    # scale: peak_ratio around 3~8, band_ratio around 0.15~0.45
    pr = np.clip((peak_ratio - min_peak_ratio) / max(1e-6, (8.0 - min_peak_ratio)), 0.0, 1.0)
    br = np.clip((band_ratio - min_band_ratio) / max(1e-6, (0.45 - min_band_ratio)), 0.0, 1.0)
    vr = np.clip((vmax - min_max_votes) / 12.0, 0.0, 1.0)
    confidence = float(np.clip(0.45 * pr + 0.35 * br + 0.20 * vr, 0.0, 1.0))

    metrics = {
        "max_votes": vmax,
        "mean_votes": vmean,
        "peak_ratio": peak_ratio,
        "band_ratio": band_ratio,
    }
    return bool(has_breath), confidence, metrics


def detect_events_v2(s_bp, fs, f_lo=0.10, f_hi=0.90,
                     win_sec=12.0, stride_sec=2.0,
                     prom_k=0.35, min_frac=0.60, max_frac=1.60):
    N = len(s_bp)
    win = int(round(win_sec * fs))
    stride = int(round(stride_sec * fs))
    votes = np.zeros(N, dtype=np.int32)

    t0_track = []
    t_center = []

    start = 0
    while start + win <= N:
        end = start + win
        seg = s_bp[start:end]

        f0, peakiness = estimate_f0_fft(seg, fs, f_lo, f_hi)
        if not np.isfinite(f0) or f0 <= 0:
            start += stride
            continue
        T0 = 1.0 / f0

        min_dist = max(1, int(round(min_frac * T0 * fs)))
        prom = max(0.15, prom_k * np.std(seg))

        troughs, _ = find_peaks(-seg, distance=min_dist, prominence=prom)
        troughs = merge_close_events(troughs, fs, T0, min_frac=min_frac)
        troughs = insert_missing_events(troughs, fs, T0, max_frac=max_frac)

        for idx in troughs:
            gi = start + int(idx)
            if 0 <= gi < N:
                votes[gi] += 2
                if gi - 1 >= 0:
                    votes[gi - 1] += 1
                if gi + 1 < N:
                    votes[gi + 1] += 1

        t0_track.append(T0)
        t_center.append((start + end) / 2.0 / fs)

        start += stride

    v_prom = max(2, int(np.percentile(votes, 95))) if votes.max() > 0 else 2
    ev, _ = find_peaks(votes.astype(np.float64),
                       distance=int(round(0.5 * fs)),
                       prominence=v_prom)

    if len(t0_track) > 0:
        T0_med = float(np.median(t0_track))
        ev = merge_close_events(ev, fs, T0_med, min_frac=min_frac)
        ev = insert_missing_events(ev, fs, T0_med, max_frac=max_frac)
    else:
        T0_med = np.nan

    return ev.astype(int), votes, (np.array(t_center), np.array(t0_track)), T0_med


def save_plot(out_png, t, s_bp, ev_idx, votes, t0_pack, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(t, s_bp, linewidth=1)
    if len(ev_idx) > 0:
        plt.scatter(t[ev_idx], s_bp[ev_idx], s=20)
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel("bandpassed signal")

    if votes is not None and votes.max() > 0:
        v = votes.astype(np.float64)
        v = v / (v.max() + 1e-9)
        v = (v - 0.5) * (np.std(s_bp) * 0.8)
        plt.plot(t, v, linewidth=1)

    if t0_pack is not None:
        t_center, t0_track = t0_pack
        if len(t0_track) > 0:
            T0_med = float(np.median(t0_track))
            plt.text(0.02, 0.95, f"T0_med={T0_med:.2f}s  (~{60.0/T0_med:.1f} bpm)",
                     transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def run_one(h5_path: Path, out_dir: Path, ds="DS1", dt=0.035,
            f_lo=0.10, f_hi=0.90, roi_topk=0.05,
            win_sec=12.0, stride_sec=2.0,
            prom_k=0.35, min_frac=0.60, max_frac=1.60,
            drop_head_sec=1.5, use_sec=None,
            gate_min_max_votes=6, gate_min_peak_ratio=3.0, gate_min_band_ratio=0.15):
    fs = 1.0 / dt
    s_raw, s_bp, src, N, meta = build_roi_waveform(
        h5_path, ds_name=ds, label_name="LABEL", fs=fs,
        f_lo=f_lo, f_hi=f_hi, roi_topk=roi_topk,
        drop_head_sec=drop_head_sec, use_sec=use_sec
    )
    t = np.arange(N) / fs

    ev_idx, votes, t0_pack, T0_med = detect_events_v2(
        s_bp, fs, f_lo, f_hi,
        win_sec=win_sec, stride_sec=stride_sec,
        prom_k=prom_k,
        min_frac=min_frac, max_frac=max_frac
    )

    has_breath, conf, gmet = breath_presence_gate(
        s_raw, votes, fs, f_lo, f_hi,
        min_max_votes=gate_min_max_votes,
        min_peak_ratio=gate_min_peak_ratio,
        min_band_ratio=gate_min_band_ratio
    )

    ev_t = ev_idx / fs
    intervals = np.diff(ev_t) if len(ev_t) >= 2 else np.array([])

    df = pd.DataFrame({
        "event_idx": ev_idx.astype(int),
        "t_s": ev_t.astype(float),
    })
    if len(df) > 1:
        df["dt_s"] = np.r_[np.nan, intervals]
        df["bpm_from_dt"] = np.r_[np.nan, 60.0 / intervals]
    else:
        df["dt_s"] = np.nan
        df["bpm_from_dt"] = np.nan

    out_dir.mkdir(parents=True, exist_ok=True)
    base = h5_path.stem

    out_csv = out_dir / f"{base}_events.csv"
    df.to_csv(out_csv, index=False, float_format="%.6f")

    out_png = out_dir / f"{base}_plot.png"
    title = f"{base} | src={src} | events={len(ev_idx)}"
    save_plot(out_png, t, s_bp, ev_idx, votes, t0_pack, title)

    est_bpm = np.nan
    bpm_out = np.nan
    if len(ev_t) >= 3:
        med_dt = float(np.median(np.diff(ev_t)))
        if med_dt > 0:
            est_bpm = 60.0 / med_dt

    if has_breath and np.isfinite(est_bpm):
        bpm_out = est_bpm

    return {
        "file": base,
        "src": src,
        "duration_s": float(N / fs),
        "events": int(len(ev_idx)),
        "T0_med_s": float(T0_med) if np.isfinite(T0_med) else np.nan,
        "est_bpm": float(est_bpm) if np.isfinite(est_bpm) else np.nan,
        "bpm_out": float(bpm_out) if np.isfinite(bpm_out) else np.nan,
        "has_breath": bool(has_breath),
        "confidence": float(conf),
        "max_votes": float(gmet.get('max_votes', np.nan)),
        "mean_votes": float(gmet.get('mean_votes', np.nan)),
        "peak_ratio": float(gmet.get('peak_ratio', np.nan)),
        "band_ratio": float(gmet.get('band_ratio', np.nan)),
        "crop_start_idx": meta["crop_start_idx"],
        "crop_end_idx": meta["crop_end_idx"],
        "crop_len": meta["crop_len"],
        "out_csv": str(out_csv),
        "out_png": str(out_png),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing .h5 files")
    ap.add_argument("--out_dir", required=True, help="Folder to save outputs")
    ap.add_argument("--recursive", action="store_true", help="Search .h5 recursively")

    ap.add_argument("--ds", default="DS1")
    ap.add_argument("--dt", type=float, default=0.035)
    ap.add_argument("--f_lo", type=float, default=0.10)
    ap.add_argument("--f_hi", type=float, default=0.90)
    ap.add_argument("--roi_topk", type=float, default=0.05)

    ap.add_argument("--win_sec", type=float, default=12.0)
    ap.add_argument("--stride_sec", type=float, default=2.0)
    ap.add_argument("--prom_k", type=float, default=0.35)
    ap.add_argument("--min_frac", type=float, default=0.60)
    ap.add_argument("--max_frac", type=float, default=1.60)

    
    # Breathing presence gate (to avoid outputting BPM on empty/background-only segments)
    ap.add_argument("--gate_min_max_votes", type=float, default=6,
                    help="Gate: minimum max(votes) to accept breathing (default 6)")
    ap.add_argument("--gate_min_peak_ratio", type=float, default=3.0,
                    help="Gate: minimum max(votes)/mean(votes) to accept breathing (default 3.0)")
    ap.add_argument("--gate_min_band_ratio", type=float, default=0.15,
                    help="Gate: minimum band_energy/total_energy ratio to accept breathing (default 0.15)")

# LABEL cropping controls
    ap.add_argument("--drop_head_sec", type=float, default=1.5,
                    help="Drop first seconds after LABEL turns 1 to avoid transient")
    ap.add_argument("--use_sec", type=float, default=None,
                    help="Use fixed seconds after crop start (default: use all remaining)")

    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.h5" if args.recursive else "*.h5"
    files = sorted(in_dir.glob(pattern))
    if not files:
        print(f"[WARN] No .h5 found in {in_dir} (recursive={args.recursive})")
        sys.exit(0)

    rows = []
    ok = 0
    fail = 0
    for p in files:
        try:
            r = run_one(
                p, out_dir,
                ds=args.ds, dt=args.dt,
                f_lo=args.f_lo, f_hi=args.f_hi,
                roi_topk=args.roi_topk,
                win_sec=args.win_sec, stride_sec=args.stride_sec,
                prom_k=args.prom_k,
                min_frac=args.min_frac, max_frac=args.max_frac,
                drop_head_sec=args.drop_head_sec, use_sec=args.use_sec,
                gate_min_max_votes=args.gate_min_max_votes,
                gate_min_peak_ratio=args.gate_min_peak_ratio,
                gate_min_band_ratio=args.gate_min_band_ratio
            )
            rows.append(r)
            ok += 1
            print(f"[OK] {p.name} -> has_breath={r['has_breath']} conf={r['confidence']:.2f} events={r['events']}  bpm_out={r['bpm_out'] if np.isfinite(r['bpm_out']) else float('nan'):.2f} est_bpm~{r['est_bpm'] if np.isfinite(r['est_bpm']) else float('nan'):.2f}  peakR={r['peak_ratio']:.2f} bandR={r['band_ratio']:.2f}  crop={r['crop_len']}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {p.name}: {e}")

    summary = pd.DataFrame(rows)
    summary_csv = out_dir / "summary.csv"
    summary.to_csv(summary_csv, index=False, float_format="%.6f")

    print(f"\n[DONE] success={ok}, fail={fail}")
    print(f"[DONE] summary saved: {summary_csv}")


if __name__ == "__main__":
    main()
