from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

from gui import BreathingMonitorApp
from engine import TGCnnProcessor

from KKT_Module import kgl
from KKT_Module.DataReceive.Core import Results
from KKT_Module.DataReceive.DataReceiver import MultiResult4168BReceiver
from KKT_Module.FiniteReceiverMachine import FRM
from KKT_Module.GuiUpdater.GuiUpdater import Updater
from KKT_Module.SettingProcess.SettingConfig import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc

DEFAULT_STREAM_REG_ADDR = 0x50000504
DEFAULT_STREAM_REG_BIT = 5
DEFAULT_STREAM_REG_VAL = 1
DEFAULT_SETTING_DIR = "K60168-Test-00256-008-v0.0.8-20230717_60cm"


class CnnInferenceUpdater(Updater):
    def __init__(self, processor: TGCnnProcessor):
        super().__init__()
        self.processor = processor
        self.n_frames = 0

    @staticmethod
    def to_frame(arr) -> np.ndarray:
        x = np.asarray(arr)
        if x.shape == (2, 32, 32):
            pass
        elif x.shape == (32, 32, 2):
            x = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected feature_map shape: {x.shape}")
        return x.astype(np.float32, copy=False)

    def update(self, res: Results):
        try:
            arr = res["feature_map"].data
            frame = self.to_frame(arr)
            self.n_frames += 1
            out = self.processor.process_frame(frame)
            if out is not None and self.n_frames % 20 == 0:
                print(
                    f"[CNN] status={out['status']} feedback={out.get('feedback_status', out['status'])} "
                    f"bpm={out['bpm']} conf={out['confidence']:.3f} raw={out.get('raw_prob', 0.0):.3f}"
                )
        except Exception as e:
            print(f"[WARN] updater failed: {e}")


class KKTController:
    def __init__(self, setting_dir: str):
        self.setting_dir = str(setting_dir)
        self.receiver = None
        self.updater = None
        self.started = False

    def connect_and_setup(self):
        print("[INFO] kgl.setLib()")
        kgl.setLib()

        print("[INFO] connectDevice()")
        dev = kgl.ksoclib.connectDevice()
        print(f"[INFO] device = {dev}")

        print("[INFO] running setting script")
        ksp = SettingProc()
        cfg = SettingConfigs()
        cfg.Chip_ID = kgl.ksoclib.getChipID().split(" ")[0]
        cfg.Processes = [
            "Reset Device",
            "Gen Process Script",
            "Gen Param Dict",
            "Get Gesture Dict",
            "Set Script",
            "Run SIC",
            "Phase Calibration",
            "Modulation On",
        ]
        cfg.setScriptDir(self.setting_dir)
        ksp.startUp(cfg)

        print("[INFO] select feature_map output")
        kgl.ksoclib.writeReg(
            DEFAULT_STREAM_REG_VAL,
            DEFAULT_STREAM_REG_ADDR,
            DEFAULT_STREAM_REG_BIT,
            DEFAULT_STREAM_REG_BIT,
            0,
        )

    def start_stream(self, updater: Updater):
        self.updater = updater
        self.receiver = MultiResult4168BReceiver()
        if hasattr(self.receiver, "actions"):
            self.receiver.actions = 1
        if hasattr(self.receiver, "rbank_ch_enable"):
            self.receiver.rbank_ch_enable = 7
        if hasattr(self.receiver, "read_interrupt"):
            self.receiver.read_interrupt = 0
        if hasattr(self.receiver, "clear_interrupt"):
            self.receiver.clear_interrupt = 0

        FRM.setReceiver(self.receiver)
        FRM.setUpdater(self.updater)
        FRM.trigger()
        FRM.start()
        self.started = True
        print("[INFO] KKT FRM started")

    def stop(self):
        try:
            if self.started and hasattr(FRM, "stop"):
                FRM.stop()
        except Exception:
            pass
        try:
            if hasattr(kgl.ksoclib, "closeCyDevice"):
                kgl.ksoclib.closeCyDevice()
        except Exception:
            pass
        self.started = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="breathclf_best.pt")
    ap.add_argument("--setting-dir", default=DEFAULT_SETTING_DIR, help="KKT setting folder path")
    ap.add_argument("--dt", type=float, default=0.035, help="frame period in seconds")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--ema-alpha", type=float, default=0.35)
    ap.add_argument("--update-sec", type=float, default=1)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    app = BreathingMonitorApp()
    controller = KKTController(args.setting_dir)
    processor = TGCnnProcessor(
        app,
        ckpt_path=ckpt_path,
        dt=args.dt,
        threshold=args.threshold,
        ema_alpha=args.ema_alpha,
        update_sec=args.update_sec,
    )
    updater = CnnInferenceUpdater(processor)

    try:
        controller.connect_and_setup()
        controller.start_stream(updater)
    except Exception as e:
        traceback.print_exc()
        print(f"[FATAL] radar init failed: {e}")
        try:
            processor.finish()
        except Exception:
            pass

    original_on_close = app.on_close

    def wrapped_close():
        try:
            controller.stop()
        finally:
            try:
                processor.finish()
            except Exception:
                pass
            original_on_close()

    app.on_close = wrapped_close
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
