from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from scipy.signal import butter, find_peaks, sosfiltfilt

from model import BreathCNN


class H5FrameSource:
    """把 .h5 當成即時串流來源。DS1 shape 預期為 (2,32,32,N)。"""

    def __init__(self, h5_path: str | Path, dt: float = 0.035, ds_name: str = "DS1"):
        self.h5_path = str(h5_path)
        self.dt = float(dt)
        self.ds_name = ds_name
        self.arr = None
        self.idx = 0
        self.last_emit = None

    def start(self):
        with h5py.File(self.h5_path, "r") as f:
            self.arr = np.asarray(f[self.ds_name][()], dtype=np.float32)
        if self.arr.ndim != 4 or self.arr.shape[:3] != (2, 32, 32):
            raise ValueError(f"Unexpected DS1 shape: {self.arr.shape}, need (2,32,32,N)")
        self.idx = 0
        self.last_emit = None

    def stop(self):
        self.arr = None

    def get_next_frame(self):
        if self.arr is None or self.idx >= self.arr.shape[-1]:
            return None
        now = time.time()
        if self.last_emit is not None:
            sleep_sec = self.dt - (now - self.last_emit)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        frame = self.arr[:, :, :, self.idx].astype(np.float32, copy=False)
        self.idx += 1
        self.last_emit = time.time()
        return frame


class TGCnnProcessor:
    """吃進單幀 (2,32,32)，產生給 tg.py queue 的即時結果。"""

    def __init__(
        self,
        app,
        ckpt_path: str | Path,
        dt: float = 0.035,
        f_lo: float = 0.10,
        f_hi: float = 0.90,
        roi_topk: float = 0.05,
        threshold: float = 0.18,
        display_threshold: float = 0.10,
        ema_alpha: float = 0.55,
        buf_sec: float = 10.0,
        bpm_window_sec: float = 6.0,
        update_sec: float = 0.6,
        calibrate_sec: float = 3.0,
    ):
        self.app = app
        self.dt = float(dt)
        self.fs = 1.0 / self.dt
        self.f_lo = float(f_lo)
        self.f_hi = float(f_hi)
        self.roi_topk = float(roi_topk)
        self.threshold = float(threshold)
        self.display_threshold = float(display_threshold)
        self.ema_alpha = float(ema_alpha)
        self.update_sec = float(update_sec)
        self.buf_len = max(96, int(round(buf_sec * self.fs)))
        self.bpm_window_len = max(128, int(round(bpm_window_sec * self.fs)))
        
        self.display_status = "UNCERTAIN"
        self.feedback_status = "UNCERTAIN"

        self.breathing_on_count = 0
        self.uncertain_on_count = 0
        self.feedback_breathing_count = 0
        self.feedback_uncertain_count = 0

        self.enter_breathing_needed = 1
        self.enter_uncertain_needed = 10
        self.feedback_enter_needed = 1
        self.feedback_leave_needed = 10

        self.flat_std_th = 0.006
        self.min_good_intervals = 1
        self.min_w_mean = 0.18
        self.last_bpm = None
        self.bpm_alpha = 0.35
        self.max_bpm_step = 6.0
        self.min_bpm = 6.0
        self.max_bpm = 30.0

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        self.T = int(ckpt["T"])
        self.model = BreathCNN(self.T)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.rdi_buf = deque(maxlen=self.buf_len)
        self.phd_buf = deque(maxlen=self.buf_len)
        self.last_emit = 0.0
        self.last_prob = None
        self.calibrate_frames = max(20, int(round(calibrate_sec * self.fs)))
        self.frame_count = 0

        self.sos = butter(
            4,
            [self.f_lo / (0.5 * self.fs), self.f_hi / (0.5 * self.fs)],
            btype="bandpass",
            output="sos",
        )

    def _robust_z(self, x: np.ndarray) -> np.ndarray:
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-9
        return (x - med) / (1.4826 * mad)

    def _rfft_bandpower(self, px_t: np.ndarray) -> np.ndarray:
        T = px_t.shape[1]
        X = np.fft.rfft(px_t, axis=1)
        Pxx = (np.abs(X) ** 2) / max(1, T)
        freqs = np.fft.rfftfreq(T, d=1.0 / self.fs)
        idx = np.where((freqs >= self.f_lo) & (freqs <= self.f_hi))[0]
        if idx.size == 0:
            return np.zeros((px_t.shape[0],), dtype=np.float64)
        return Pxx[:, idx].sum(axis=1)

    def _peakiness_per_px(self, px_t: np.ndarray) -> np.ndarray:
        T = px_t.shape[1]
        X = np.fft.rfft(px_t - px_t.mean(axis=1, keepdims=True), axis=1)
        Pxx = (np.abs(X) ** 2) / max(1, T)
        freqs = np.fft.rfftfreq(T, d=1.0 / self.fs)
        band = (freqs >= self.f_lo) & (freqs <= self.f_hi)
        if not np.any(band):
            return np.ones((px_t.shape[0],), dtype=np.float64)
        Pb = Pxx[:, band]
        peak = Pb.max(axis=1)
        med = np.median(Pb, axis=1) + 1e-12
        return peak / med

    def _build_roi_waveform_from_buffer(self) -> Optional[np.ndarray]:
        n = len(self.rdi_buf)
        if n < self.T:
            return None

        rdi = np.stack(self.rdi_buf, axis=-1).astype(np.float32, copy=False)
        phd = np.stack(self.phd_buf, axis=-1).astype(np.float32, copy=False)
        H, W, N = rdi.shape
        P = H * W

        rdi_px = rdi.reshape(P, N)
        phd_px = phd.reshape(P, N)

        bp_rdi = self._rfft_bandpower(rdi_px)
        bp_phd = self._rfft_bandpower(phd_px)
        pk_rdi = self._peakiness_per_px(rdi_px.astype(np.float64))
        pk_phd = self._peakiness_per_px(phd_px.astype(np.float64))

        score_rdi = bp_rdi * np.clip(pk_rdi, 1.0, 10.0)
        score_phd = bp_phd * np.clip(pk_phd, 1.0, 10.0)

        k = max(1, int(P * self.roi_topk))
        top_rdi = np.partition(score_rdi, -k)[-k:].mean()
        top_phd = np.partition(score_phd, -k)[-k:].mean()

        use_rdi = bool(top_rdi >= top_phd)
        score_use = score_rdi if use_rdi else score_phd
        roi_idx = np.argpartition(score_use, -k)[-k:]
        raw = (rdi_px if use_rdi else phd_px)[roi_idx].mean(axis=0).astype(np.float64, copy=False)
        raw = raw - np.median(raw)
        z = self._robust_z(raw)

        if len(z) < 16:
            return None
        try:
            padlen = min(len(z) - 1, 3 * (max(len(self.sos), 1) - 1))
            if padlen >= 1:
                bp = sosfiltfilt(self.sos, z, padlen=padlen)
            else:
                bp = sosfiltfilt(self.sos, z)
        except Exception:
            return None
        return bp.astype(np.float32, copy=False)

    def _cnn_confidence(self, wave_bp: np.ndarray) -> tuple[float, int]:
        seg = np.asarray(wave_bp[-self.T :], dtype=np.float32)
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) + 1e-6
        seg = (seg - med) / (1.4826 * mad)
        x = torch.from_numpy(seg).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        prob = float(probs[1])
        pred = int(np.argmax(probs))
        return prob, pred

    def _estimate_bpm(self, wave_bp: np.ndarray) -> tuple[Optional[float], int, float]:
        if len(wave_bp) < self.bpm_window_len:
            print("[BPM] too short", len(wave_bp), self.bpm_window_len)
            return None, 0, 0.0

        x = np.asarray(wave_bp[-self.bpm_window_len:], dtype=np.float64)
        x = x - np.mean(x)
        std = float(np.std(x))
        if std < 1e-6:
            print("[BPM] std too small", std)
            return None, 0, 0.0

        min_bpm = 4.0
        max_bpm = 35.0

        min_dist = max(1, int(round((60.0 / max_bpm) * self.fs)))
        prom = max(0.22 * std, 0.08)

        peaks, props = find_peaks(x, distance=min_dist, prominence=prom)
        print(f"[BPM] std={std:.4f} prom={prom:.4f} peaks={len(peaks)}")

        if len(peaks) < 2:
            return None, int(len(peaks)), 0.0

        intervals = np.diff(peaks) / self.fs
        good = intervals[(intervals >= 60.0 / max_bpm) & (intervals <= 60.0 / min_bpm)]
        print(f"[BPM] intervals={intervals} good={good}")

        if len(good) < 1:
            return None, int(len(good)), 0.0

        bpm = 60.0 / float(np.median(good))
        bpm = float(np.clip(bpm, min_bpm, max_bpm))

        prom_arr = props.get("prominences", np.ones(len(peaks), dtype=np.float64))
        w_mean = float(np.mean(prom_arr)) / (std + 1e-6)

        print(f"[BPM] bpm={bpm:.2f} good_count={len(good)} w_mean={w_mean:.3f}")
        return bpm, int(len(good)), w_mean

    def _push(self, data: dict):
        q = self.app.queue
        try:
            while q.full():
                q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(data)
        except queue.Full:
            pass

    def process_frame(self, frame: np.ndarray) -> Optional[dict]:
        if frame.shape != (2, 32, 32):
            return None

        self.rdi_buf.append(np.asarray(frame[0], dtype=np.float32))
        self.phd_buf.append(np.asarray(frame[1], dtype=np.float32))
        self.frame_count += 1

        now = time.time()
        if now - self.last_emit < self.update_sec:
            return None
        self.last_emit = now

        # ===== 校準期 =====
        if self.frame_count < self.calibrate_frames:
            out = {
                "bpm": None,
                "status": "CALIBRATING",
                "feedback_status": "CALIBRATING",
                "confidence": 0.0,
                "wave": [],
            }
            self._push(out)
            return out

        # ===== 建 waveform =====
        wave_bp = self._build_roi_waveform_from_buffer()
        if wave_bp is None or len(wave_bp) < self.T:
            out = {
                "bpm": None,
                "status": "UNCERTAIN",
                "feedback_status": "UNCERTAIN",
                "confidence": 0.0,
                "wave": [],
            }
            self._push(out)
            return out

        # ===== CNN confidence =====
        prob, pred = self._cnn_confidence(wave_bp)
        smooth_prob = prob if self.last_prob is None else (
            self.ema_alpha * prob + (1.0 - self.ema_alpha) * self.last_prob
        )
        self.last_prob = smooth_prob

        # ===== GUI waveform =====
        view_wave = np.asarray(wave_bp[-220:], dtype=np.float32)
        if len(view_wave) >= 3:
            kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32)
            kernel = kernel / kernel.sum()
            view_wave = np.convolve(view_wave, kernel, mode="same")

        # ===== BPM =====
        bpm, intervals_used, w_mean = self._estimate_bpm(wave_bp)
        tail_std = float(np.std(wave_bp[-120:])) if len(wave_bp) >= 120 else float(np.std(wave_bp))

        # ===== 候選狀態（瞬時）=====
        candidate_status = "BREATHING" if (
            (smooth_prob >= 0.20) and
            (bpm is not None) and
            (tail_std >= self.flat_std_th)
        ) else "UNCERTAIN"

        # 只要 confidence 或 bpm 有一邊成立，就先進 feedback breathing
        candidate_feedback = "BREATHING" if ((smooth_prob >= self.threshold) or (bpm is not None and smooth_prob >= 0.08)) else "UNCERTAIN"

        # ===== flat waveform 保護 =====
        # 波形太平通常表示遮住雷達 / 無人 / 無有效訊號
        if tail_std < self.flat_std_th:
            candidate_feedback = "UNCERTAIN"
            candidate_status = "UNCERTAIN"
            bpm = None
        # ===== display debounce =====
        if candidate_status == "BREATHING":
            self.breathing_on_count += 1
            self.uncertain_on_count = 0
        else:
            self.uncertain_on_count += 1
            self.breathing_on_count = 0

        if self.display_status != "BREATHING":
            if self.breathing_on_count >= self.enter_breathing_needed:
                self.display_status = "BREATHING"
        else:
            if self.uncertain_on_count >= self.enter_uncertain_needed:
                self.display_status = "UNCERTAIN"

        # ===== feedback debounce =====
        if candidate_feedback == "BREATHING":
            self.feedback_breathing_count += 1
            self.feedback_uncertain_count = 0
        else:
            self.feedback_uncertain_count += 1
            self.feedback_breathing_count = 0

        if self.feedback_status != "BREATHING":
            if self.feedback_breathing_count >= self.feedback_enter_needed:
                self.feedback_status = "BREATHING"
        else:
            if self.feedback_uncertain_count >= self.feedback_leave_needed:
                self.feedback_status = "UNCERTAIN"

        # ===== 最終輸出狀態 =====
        status = self.display_status
        feedback_status = self.feedback_status
        if status == "BREATHING" and bpm is None:
            bpm = self.last_bpm

        if bpm is not None:
            bpm = float(np.clip(bpm, self.min_bpm, self.max_bpm))
            if self.last_bpm is not None:
                delta = bpm - self.last_bpm
                delta = float(np.clip(delta, -self.max_bpm_step, self.max_bpm_step))
                bpm = self.last_bpm + delta
                bpm = self.bpm_alpha * bpm + (1.0 - self.bpm_alpha) * self.last_bpm
            self.last_bpm = bpm

        out_bpm = bpm if status == "BREATHING" else None

        out = {
            "bpm": out_bpm,
            "status": status,
            "feedback_status": feedback_status,
            "confidence": float(smooth_prob),
            "wave": view_wave.tolist(),
            "intervals_used": int(intervals_used),
            "w_mean": float(w_mean),
            "pred": int(pred),
            "raw_prob": float(prob),
            "tail_std": float(tail_std),
        }

        print(
            f"[CNN] status={status} feedback={feedback_status} "
            f"bpm={out_bpm} conf={smooth_prob:.3f} raw={prob:.3f} "
            f"tail_std={tail_std:.5f}"
        )

        self._push(out)
        return out
    def finish(self):
        out = {"bpm": None, "status": "UNCERTAIN", "feedback_status": "UNCERTAIN", "confidence": 0.0, "wave": []}
        self._push(out)
        return out


class TGCnnEngine:
    """輪詢型來源用的包裝器，例如 .h5 模擬串流。"""

    def __init__(self, app, frame_source, ckpt_path: str | Path, **kwargs):
        self.app = app
        self.frame_source = frame_source
        self.processor = TGCnnProcessor(app, ckpt_path, **kwargs)
        self.running = False
        self.thread = None

    def start(self):
        self.frame_source.start()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        try:
            self.frame_source.stop()
        except Exception:
            pass

    def _loop(self):
        while self.running and not self.app.stop_event.is_set():
            frame = self.frame_source.get_next_frame()
            if frame is None:
                break
            try:
                self.processor.process_frame(frame)
            except Exception as e:
                print(f"[WARN] processor failed: {e}")
        self.processor.finish()
