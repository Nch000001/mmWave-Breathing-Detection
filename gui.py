#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import math
import os
import queue
import random
import threading
import time
from collections import deque

import customtkinter as ctk
from tkinter import filedialog, messagebox

try:
    import winsound
    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==============================
# Demo 固定配色（直接改這裡 Hex 即可）
# Level 3 的 Pain Index 會固定紅色，因此呼吸燈不使用紅色
# ==============================
BREATH_LIGHT_IDLE = "#2B2B2B"
BREATH_LIGHT_LEVEL1_A = "#1E58A9"
BREATH_LIGHT_LEVEL1_B = "#893000"
BREATH_LIGHT_LEVEL2_A = "#D44A18"
BREATH_LIGHT_LEVEL2_B = "#008179"
PAIN_COLOR_L0 = "#AAA7A7"
PAIN_COLOR_L1 = "#2ECC71"
PAIN_COLOR_L2 = "#F1C40F"
PAIN_COLOR_L3 = "#E74C3C"

# ==============================
# VLC 初始化
# ==============================
HAS_VLC = False
VLC_STATUS = "VLC 未初始化"
vlc = None


def setup_vlc():
    global HAS_VLC, VLC_STATUS, vlc

    candidate_dirs = [
        r"C:\Program Files\VideoLAN\VLC",
    ]

    chosen_dir = None
    for d in candidate_dirs:
        if os.path.isfile(os.path.join(d, "libvlc.dll")):
            chosen_dir = d
            break

    try:
        if chosen_dir and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(chosen_dir)
            os.environ["PATH"] = chosen_dir + os.pathsep + os.environ.get("PATH", "")

        import vlc as _vlc
        _ = _vlc.Instance("--no-video")
        vlc = _vlc
        HAS_VLC = True
        VLC_STATUS = f"VLC 可用：{chosen_dir}" if chosen_dir else "VLC 可用"
    except Exception as e:
        HAS_VLC = False
        vlc = None
        VLC_STATUS = f"VLC 載入失敗：{e}"


setup_vlc()


class PlaylistPlayer:
    """播放清單播放器：用於 Level 1 舒緩音樂。"""

    def __init__(self):
        self.files = []
        self.current_index = -1
        self.is_playing = False
        self.instance = None
        self.player = None
        self.last_status = VLC_STATUS

        if HAS_VLC and vlc is not None:
            try:
                self.instance = vlc.Instance("--no-video")
                self.player = self.instance.media_player_new()
                self.last_status = "VLC 就緒"
            except Exception as e:
                self.instance = None
                self.player = None
                self.last_status = f"VLC 建立播放器失敗：{e}"

    def add_files(self, paths):
        for p in paths:
            if p and p not in self.files:
                self.files.append(p)
        if self.current_index < 0 and self.files:
            self.current_index = 0

    def clear(self):
        self.stop()
        self.files = []
        self.current_index = -1

    def current_file(self):
        if 0 <= self.current_index < len(self.files):
            return self.files[self.current_index]
        return None

    def play_current(self):
        cur = self.current_file()
        if not cur:
            return False, "播放清單為空"
        if not os.path.exists(cur):
            return False, f"找不到檔案：{cur}"
        if not HAS_VLC or self.player is None or self.instance is None:
            return False, self.last_status
        try:
            media = self.instance.media_new(cur)
            media.add_option(":no-video")
            self.player.set_media(media)
            self.player.play()
            self.is_playing = True
            self.last_status = f"播放中：{os.path.basename(cur)}"
            return True, self.last_status
        except Exception as e:
            self.is_playing = False
            self.last_status = f"播放失敗：{e}"
            return False, self.last_status

    def toggle(self):
        if not HAS_VLC or self.player is None:
            return False, self.last_status
        try:
            if self.player.is_playing():
                self.player.stop()
                self.is_playing = False
                self.last_status = "已暫停"
                return True, self.last_status
            ok, msg = self.play_current()
            if ok:
                self.is_playing = True
            return ok, msg
        except Exception as e:
            self.last_status = f"切換播放失敗：{e}"
            return False, self.last_status

    def auto_start(self):
        if self.is_playing:
            return True, self.last_status
        return self.play_current()

    def stop(self):
        if HAS_VLC and self.player is not None:
            try:
                self.player.stop()
            except Exception:
                pass
        self.is_playing = False

    def next(self):
        if not self.files:
            return False, "播放清單為空"
        self.current_index = (self.current_index + 1) % len(self.files)
        return self.play_current()

    def prev(self):
        if not self.files:
            return False, "播放清單為空"
        self.current_index = (self.current_index - 1) % len(self.files)
        return self.play_current()


class SingleAudioPlayer:
    """單一音檔播放器：用於安撫語音或警報音。"""

    def __init__(self, name="audio"):
        self.name = name
        self.instance = None
        self.player = None
        self.path = ""
        self.last_status = "未設定"

        if HAS_VLC and vlc is not None:
            try:
                self.instance = vlc.Instance("--no-video")
                self.player = self.instance.media_player_new()
                self.last_status = f"{name} 就緒"
            except Exception as e:
                self.instance = None
                self.player = None
                self.last_status = f"{name} VLC 建立失敗：{e}"

    def set_file(self, path):
        self.path = path or ""
        self.last_status = os.path.basename(self.path) if self.path else "未設定"

    def play(self):
        if not self.path:
            return False, f"{self.name} 未設定音檔"
        if not os.path.exists(self.path):
            return False, f"找不到檔案：{self.path}"
        if not HAS_VLC or self.player is None or self.instance is None:
            return False, f"{self.name} VLC 不可用"
        try:
            media = self.instance.media_new(self.path)
            media.add_option(":no-video")
            self.player.set_media(media)
            self.player.play()
            self.last_status = f"{self.name} 播放中：{os.path.basename(self.path)}"
            return True, self.last_status
        except Exception as e:
            self.last_status = f"{self.name} 播放失敗：{e}"
            return False, self.last_status

    def stop(self):
        if self.player is not None:
            try:
                self.player.stop()
            except Exception:
                pass

    def is_playing(self):
        try:
            return self.player is not None and bool(self.player.is_playing())
        except Exception:
            return False


class BeepAlertPlayer:
    """Level 3 退而求其次的刺耳警報聲。"""

    def __init__(self):
        self.running = False
        self.thread = None
        self.last_status = "警報待命"

    def start(self):
        if self.running:
            return True, self.last_status
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.last_status = "警報音啟動"
        return True, self.last_status

    def _loop(self):
        while self.running:
            try:
                if HAS_WINSOUND:
                    winsound.Beep(1800, 350)
                    time.sleep(0.15)
                    winsound.Beep(1200, 350)
                else:
                    time.sleep(0.7)
            except Exception:
                time.sleep(0.7)

    def stop(self):
        self.running = False
        self.last_status = "警報待命"


class CustomSettingsPopup(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("設定")
        self.geometry("540x520")
        self.transient(parent)
        self.focus_set()
        self._set_centered()

        self.scroll_frame = ctk.CTkScrollableFrame(self, width=480, height=430)
        self.scroll_frame.pack(fill="both", expand=True, padx=14, pady=14)

        ctk.CTkLabel(
            self.scroll_frame,
            text="設定",
            font=("Microsoft JhengHei", 18, "bold")
        ).pack(pady=(8, 18))

        switch_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        switch_frame.pack(fill="x", padx=8, pady=(0, 8))

        self.switch_tts = ctk.CTkSwitch(switch_frame, text="啟用 Level 2 安撫語音")
        if self.parent.enable_tts:
            self.switch_tts.select()
        else:
            self.switch_tts.deselect()
        self.switch_tts.pack(pady=6, anchor="w")

        self.switch_music_auto = ctk.CTkSwitch(switch_frame, text="Level 1 自動播放音樂")
        if self.parent.enable_auto_music:
            self.switch_music_auto.select()
        else:
            self.switch_music_auto.deselect()
        self.switch_music_auto.pack(pady=6, anchor="w")

        ctk.CTkLabel(self.scroll_frame, text="緊急聯絡人資訊 (Line ID / 電話)", font=("Microsoft JhengHei", 14)).pack(pady=(16, 0))
        self.entry_contact = ctk.CTkEntry(self.scroll_frame, width=380)
        self.entry_contact.insert(0, self.parent.contact_info)
        self.entry_contact.pack(pady=5)

        ctk.CTkLabel(self.scroll_frame, text="自訂音樂名稱", font=("Microsoft JhengHei", 14)).pack(pady=(16, 0))
        self.entry_music = ctk.CTkEntry(self.scroll_frame, width=380)
        self.entry_music.insert(0, self.parent.current_music_title)
        self.entry_music.pack(pady=5)

        ctk.CTkLabel(self.scroll_frame, text="Pain Index 分級門檻", font=("Microsoft JhengHei", 14, "bold")).pack(pady=(18, 0))
        grid = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        grid.pack(pady=8)
        ctk.CTkLabel(grid, text="Level 1 起始").grid(row=0, column=0, padx=6, pady=4)
        ctk.CTkLabel(grid, text="Level 2 起始").grid(row=0, column=1, padx=6, pady=4)
        ctk.CTkLabel(grid, text="Level 3 起始").grid(row=0, column=2, padx=6, pady=4)

        self.entry_l1 = ctk.CTkEntry(grid, width=90)
        self.entry_l2 = ctk.CTkEntry(grid, width=90)
        self.entry_l3 = ctk.CTkEntry(grid, width=90)
        self.entry_l1.insert(0, str(self.parent.level1_threshold))
        self.entry_l2.insert(0, str(self.parent.level2_threshold))
        self.entry_l3.insert(0, str(self.parent.level3_threshold))
        self.entry_l1.grid(row=1, column=0, padx=6, pady=4)
        self.entry_l2.grid(row=1, column=1, padx=6, pady=4)
        self.entry_l3.grid(row=1, column=2, padx=6, pady=4)

        ctk.CTkLabel(self.scroll_frame, text="Level 2 建議呼吸頻率 BPM", font=("Microsoft JhengHei", 14)).pack(pady=(16, 0))
        self.entry_guided_bpm = ctk.CTkEntry(self.scroll_frame, width=140)
        self.entry_guided_bpm.insert(0, str(self.parent.guided_bpm))
        self.entry_guided_bpm.pack(pady=5)

        # ctk.CTkLabel(self.scroll_frame, text="Level 2 開場音檔", font=("Microsoft JhengHei", 14)).pack(pady=(18, 0))
        # row_intro = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        # row_intro.pack(pady=5)
        # self.entry_intro = ctk.CTkEntry(row_intro, width=320)
        # self.entry_intro.insert(0, self.parent.level2_intro_file)
        # self.entry_intro.pack(side="left", padx=6)
        # ctk.CTkButton(row_intro, text="選擇", width=70, command=lambda: self.pick_audio_file(self.entry_intro, "選擇 Level 2 開場音檔")).pack(side="left", padx=4)

        # ctk.CTkLabel(self.scroll_frame, text="吸氣音檔", font=("Microsoft JhengHei", 14)).pack(pady=(12, 0))
        # row_inhale = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        # row_inhale.pack(pady=5)
        # self.entry_inhale = ctk.CTkEntry(row_inhale, width=320)
        # self.entry_inhale.insert(0, self.parent.inhale_file)
        # self.entry_inhale.pack(side="left", padx=6)
        # ctk.CTkButton(row_inhale, text="選擇", width=70, command=lambda: self.pick_audio_file(self.entry_inhale, "選擇吸氣音檔")).pack(side="left", padx=4)

        # ctk.CTkLabel(self.scroll_frame, text="吐氣音檔", font=("Microsoft JhengHei", 14)).pack(pady=(12, 0))
        # row_exhale = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        # row_exhale.pack(pady=5)
        # self.entry_exhale = ctk.CTkEntry(row_exhale, width=320)
        # self.entry_exhale.insert(0, self.parent.exhale_file)
        # self.entry_exhale.pack(side="left", padx=6)
        # ctk.CTkButton(row_exhale, text="選擇", width=70, command=lambda: self.pick_audio_file(self.entry_exhale, "選擇吐氣音檔")).pack(side="left", padx=4)

        # ctk.CTkLabel(
        #     self.scroll_frame,
        #     font=("Microsoft JhengHei", 11),
        #     text_color="gray",
        #     wraplength=420,
        #     justify="left",
        # ).pack(pady=(14, 0))

        button_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        button_frame.pack(pady=24)
        ctk.CTkButton(button_frame, text="儲存設定", command=self.save_settings).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="取消", fg_color="gray", command=self.destroy).pack(side="left", padx=10)

    def _set_centered(self):
        self.update_idletasks()
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_w = max(self.parent.winfo_width(), 1000)
        parent_h = max(self.parent.winfo_height(), 700)
        popup_w = 540
        popup_h = 520
        x = parent_x + (parent_w // 2) - (popup_w // 2)
        y = parent_y + (parent_h // 2) - (popup_h // 2)
        self.geometry(f"{popup_w}x{popup_h}+{x}+{y}")

    def pick_audio_file(self, entry_widget, title):
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Audio files", "*.mp3 *.wav *.m4a *.aac *.flac *.mp4"), ("All files", "*.*")]
        )
        if path:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, path)

    def save_settings(self):
        new_contact = self.entry_contact.get().strip()
        new_music = self.entry_music.get().strip()

        try:
            l1 = float(self.entry_l1.get().strip())
            l2 = float(self.entry_l2.get().strip())
            l3 = float(self.entry_l3.get().strip())
            guided_bpm = float(self.entry_guided_bpm.get().strip())
        except ValueError:
            messagebox.showerror("輸入錯誤", "Pain Index 門檻與建議 BPM 必須是數字")
            return

        if not (0 <= l1 < l2 < l3 <= 100):
            messagebox.showerror("輸入錯誤", "請確認門檻符合：0 <= L1 < L2 < L3 <= 100")
            return

        self.parent.enable_tts = bool(self.switch_tts.get())
        self.parent.enable_auto_music = bool(self.switch_music_auto.get())
        self.parent.update_contact_info(new_contact)
        self.parent.update_music_title(new_music)

        self.parent.level1_threshold = l1
        self.parent.level2_threshold = l2
        self.parent.level3_threshold = l3
        self.parent.guided_bpm = guided_bpm

        self.parent.save_settings_to_disk(show_popup=True)
        self.destroy()

class BreathingMonitorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("毫米波呼吸監測系統")
        self.geometry("1280x860")

        self.settings_path = os.path.join(os.path.dirname(__file__), "GUIT_settings.json")

        # 基本狀態
        self.is_playing = False
        self.contact_info = ""
        self.current_mode = "預設模式"
        self.breath_step = 0
        self.current_music_title = "舒緩引導音樂 (預設)"
        self.breath_colors = [BREATH_LIGHT_LEVEL1_A]
        self.current_bpm = None
        self.current_status = "CALIBRATING"
        self.current_confidence = 0.0

        # Pain Index 與分級
        self.bpm_history = deque(maxlen=30)
        self.current_pain_index = 100.0
        self.current_pain_level = 0
        self.current_variability = 0.0
        self.current_slope = 0.0
        self.last_feedback_level = -1
        self.display_pain = 0.0

        self.level1_threshold = 14.0
        self.level2_threshold = 24.0
        self.level3_threshold = 34.0
        self.guided_bpm = 6.0
        self.enable_tts = True
        self.enable_auto_music = True

        self.level2_intro_file = r"C:/Users/macho/Desktop/py389/final/debug/開場.m4a"
        self.inhale_file = r"C:/Users/macho/Desktop/py389/final/debug/吸氣.m4a"
        self.exhale_file = r"C:/Users/macho/Desktop/py389/final/debug/吐氣.m4a" 
        self.alert_file = " "
        self.alarm_active = False

        # Level 2 語音循環控制
        self.level2_loop_job = None
        self.level2_intro_wait_job = None
        self.level2_phase = 0   # 0=吸氣, 1=吐氣

        # 播放器
        self.player = PlaylistPlayer()
        self.level2_intro_player = SingleAudioPlayer("Level 2 開場語音")
        self.inhale_player = SingleAudioPlayer("吸氣語音")
        self.exhale_player = SingleAudioPlayer("吐氣語音")
        self.alert_player = SingleAudioPlayer("警報音")
        self.beep_alert = BeepAlertPlayer()

        self.load_settings_from_disk()
        self.level2_intro_player.set_file(self.level2_intro_file)
        self.inhale_player.set_file(self.inhale_file)
        self.exhale_player.set_file(self.exhale_file)
        self.alert_player.set_file(self.alert_file)

        self.queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()

        self._create_main_container()
        self._create_bottom_music_area()
        self._create_data_dashboard_area()

        self.apply_loaded_settings_to_ui()
        self.update_breathing_background()
        self.poll_engine()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ==============================
    # 設定存檔 / 讀檔
    # ==============================
    def load_settings_from_disk(self):
        if not os.path.exists(self.settings_path):
            return
        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.contact_info = data.get("contact_info", "")
            self.current_mode = data.get("current_mode", "預設模式")
            self.current_music_title = data.get("current_music_title", "舒緩引導音樂 (預設)")
            self.level1_threshold = float(data.get("level1_threshold", 30.0))
            self.level2_threshold = float(data.get("level2_threshold", 55.0))
            self.level3_threshold = float(data.get("level3_threshold", 75.0))
            self.guided_bpm = float(data.get("guided_bpm", 6.0))
            self.enable_tts = bool(data.get("enable_tts", True))
            self.enable_auto_music = bool(data.get("enable_auto_music", True))

            self.level2_intro_file = data.get("level2_intro_file", "")
            self.inhale_file = data.get("inhale_file", "")
            self.exhale_file = data.get("exhale_file", "")
            self.alert_file = ""

            playlist = data.get("playlist_files", [])
            if isinstance(playlist, list):
                existing = [p for p in playlist if os.path.exists(p)]
                self.player.add_files(existing)
        except Exception as e:
            print("讀取設定失敗:", e)

    def save_settings_to_disk(self, show_popup=False):
        data = {
            "contact_info": self.contact_info,
            "current_mode": self.current_mode,
            "current_music_title": self.current_music_title,
            "playlist_files": self.player.files,
            "level1_threshold": self.level1_threshold,
            "level2_threshold": self.level2_threshold,
            "level3_threshold": self.level3_threshold,
            "guided_bpm": self.guided_bpm,
            "enable_tts": self.enable_tts,
            "enable_auto_music": self.enable_auto_music,
            "level2_intro_file": self.level2_intro_file,
            "inhale_file": self.inhale_file,
            "exhale_file": self.exhale_file,
        }
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if show_popup:
                messagebox.showinfo("設定", f"已存檔\n{self.settings_path}")
        except Exception as e:
            messagebox.showerror("存檔失敗", str(e))

    def apply_loaded_settings_to_ui(self):
        self.after(50, self._apply_loaded_settings_to_ui)

    def _apply_loaded_settings_to_ui(self):
        if hasattr(self, "mode_switch"):
            self.mode_switch.set(self.current_mode)
        self.update_contact_info(self.contact_info)
        self.update_music_title(self.current_music_title)
        self.update_playlist_info()
        self.update_vlc_status()

    # ==============================
    # UI 建立
    # ==============================
    def _create_main_container(self):
        self.app_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.app_frame.pack(side="top", fill="both", expand=True)

        self.right_container = ctk.CTkFrame(self.app_frame, fg_color="transparent")
        self.right_container.pack(side="right", fill="both", expand=True, padx=20, pady=20)

    def _create_data_dashboard_area(self):
        self.dashboard_frame = ctk.CTkFrame(self.right_container, corner_radius=15)
        self.dashboard_frame.place(relx=0, rely=0, relwidth=1, relheight=0.76)

        self.breath_bg = ctk.CTkFrame(self.dashboard_frame, fg_color=self.breath_colors[0], corner_radius=15)
        self.breath_bg.place(relx=0.01, rely=0.02, relwidth=0.98, relheight=0.96)

        ctk.CTkLabel(
            self.breath_bg,
            text="Breathing Monitor",
            font=("Arial", 24)
        ).place(relx=0.5, rely=0.08, anchor="center")

        # 設定按鈕放右上角，避開主要資訊
        self.btn_advanced_settings = ctk.CTkButton(
            self.breath_bg,
            text="設定",
            width=80,
            height=32,
            fg_color="#3A3A3A",
            command=self.open_settings_popup
        )
        self.btn_advanced_settings.place(relx=0.94, rely=0.06, anchor="ne")

        self.val_bpm = ctk.CTkLabel(
            self.breath_bg,
            text="--",
            font=("Arial", 96, "bold"),
            text_color="#3498db"
        )


        self.val_bpm.place(relx=0.5, rely=0.24, anchor="center")
        self.lbl_bpm = ctk.CTkLabel(self.breath_bg, text="BPM", font=("Arial", 24))
        self.lbl_bpm.place(relx=0.5, rely=0.36, anchor="center")

        self.lbl_status = ctk.CTkLabel(
            self.breath_bg,
            text="Status: CALIBRATING",
            font=("Arial", 22)
        )
        self.lbl_status.place(relx=0.5, rely=0.46, anchor="center")

        self.lbl_level_big = ctk.CTkLabel(
            self.breath_bg,
            text="LEVEL --",
            font=("Arial", 28, "bold"),
            text_color="#BFBFBF"
        )
        self.lbl_level_big.place(relx=0.5, rely=0.54, anchor="center")

        self.lbl_feedback_summary = ctk.CTkLabel(
            self.breath_bg,
            text="系統監測中，等待 Pain Index 觸發回饋。",
            font=("Microsoft JhengHei", 18),
            wraplength=860,
            justify="center",
        )
        self.lbl_feedback_summary.place(relx=0.5, rely=0.63, anchor="center")

        self.wave_card = ctk.CTkFrame(self.breath_bg, fg_color="#1E1E1E", corner_radius=12)
        self.wave_card.place(relx=0.06, rely=0.73, relwidth=0.88, relheight=0.17)

        self.wave_title = ctk.CTkLabel(self.wave_card, text="Breathing Waveform", font=("Arial", 14))
        self.wave_title.pack(pady=(6, 0))

        self.wave_canvas = ctk.CTkCanvas(
            self.wave_card,
            width=900,
            height=95,
            bg="#1E1E1E",
            highlightthickness=0,
            bd=0
        )
        self.wave_canvas.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self.subtitle_frame = ctk.CTkFrame(self.right_container, fg_color="#1A1A1A", height=60, corner_radius=10)
        self.subtitle_frame.place(relx=0.08, rely=0.72, relwidth=0.84, relheight=0.07)

        self.lbl_subtitle = ctk.CTkLabel(
            self.subtitle_frame,
            text="系統監測中，請保持平穩呼吸...",
            font=("Microsoft JhengHei", 18)
        )
        self.lbl_subtitle.pack(pady=10)

    def _create_bottom_music_area(self):
        self.music_frame = ctk.CTkFrame(self.right_container, corner_radius=15)
        self.music_frame.place(relx=0, rely=0.81, relwidth=1, relheight=0.19)

        music_info_frame = ctk.CTkFrame(self.music_frame, fg_color="transparent")
        music_info_frame.pack(side="left", padx=20, pady=12)

        self.music_title = ctk.CTkLabel(
            music_info_frame,
            text=f"🎵 當前播放: {self.current_music_title}",
            font=("Microsoft JhengHei", 13)
        )
        self.music_title.pack(anchor="w")

        self.contact_info_label = ctk.CTkLabel(
            music_info_frame,
            text="緊急聯絡人: 尚未設定",
            font=("Microsoft JhengHei", 11),
            text_color="gray"
        )
        self.contact_info_label.pack(anchor="w")

        self.playlist_info_label = ctk.CTkLabel(
            music_info_frame,
            text="播放清單: 0 首",
            font=("Microsoft JhengHei", 11),
            text_color="gray"
        )
        self.playlist_info_label.pack(anchor="w")

        self.vlc_status_label = ctk.CTkLabel(
            music_info_frame,
            text=VLC_STATUS,
            font=("Microsoft JhengHei", 11),
            text_color="gray"
        )
        self.vlc_status_label.pack(anchor="w")

        button_frame = ctk.CTkFrame(self.music_frame, fg_color="transparent")
        button_frame.pack(side="right", padx=16, pady=12)

        self.btn_add_music = ctk.CTkButton(button_frame, text="加入音樂", width=88, command=self.add_music_files)
        self.btn_add_music.pack(side="left", padx=4)

        self.btn_prev = ctk.CTkButton(button_frame, text="上一首", width=68, fg_color="gray", command=self.prev_music)
        self.btn_prev.pack(side="left", padx=3)

        self.btn_play = ctk.CTkButton(button_frame, text="播放", width=80, command=self.music_toggle)
        self.btn_play.pack(side="left", padx=4)

        self.btn_next = ctk.CTkButton(button_frame, text="下一首", width=68, fg_color="gray", command=self.next_music)
        self.btn_next.pack(side="left", padx=3)

        self.btn_clear_music = ctk.CTkButton(button_frame, text="清空", width=68, fg_color="#8B5E3C", command=self.clear_music_files)
        self.btn_clear_music.pack(side="left", padx=3)

    # ==============================
    # 音樂 / 狀態 UI
    # ==============================
    def update_playlist_info(self):
        self.playlist_info_label.configure(text=f"播放清單: {len(self.player.files)} 首")

    def update_vlc_status(self, text=None):
        if text is None:
            text = self.player.last_status
        self.vlc_status_label.configure(text=text)

    def add_music_files(self):
        paths = filedialog.askopenfilenames(
            title="選擇音樂檔案",
            filetypes=[("Audio/Video files", "*.mp3 *.wav *.m4a *.mp4 *.aac *.flac"), ("All files", "*.*")],
        )
        if paths:
            self.player.add_files(list(paths))
            self.update_playlist_info()
            self.save_settings_to_disk(show_popup=False)
            self.update_vlc_status(f"已加入 {len(paths)} 首")

    def clear_music_files(self):
        self.player.clear()
        self.update_playlist_info()
        self.btn_play.configure(text="播放")
        self.update_vlc_status("播放清單已清空")
        self.save_settings_to_disk(show_popup=False)

    def prev_music(self):
        ok, msg = self.player.prev()
        self.is_playing = self.player.is_playing
        self.btn_play.configure(text="暫停" if self.is_playing else "播放")
        self.update_music_title_from_current_file()
        self.update_vlc_status(msg)
        if not ok:
            messagebox.showinfo("音樂播放", msg)

    def next_music(self):
        ok, msg = self.player.next()
        self.is_playing = self.player.is_playing
        self.btn_play.configure(text="暫停" if self.is_playing else "播放")
        self.update_music_title_from_current_file()
        self.update_vlc_status(msg)
        if not ok:
            messagebox.showinfo("音樂播放", msg)

    def update_music_title_from_current_file(self):
        cur = self.player.current_file()
        if cur:
            self.music_title.configure(text=f"🎵 當前播放: {os.path.basename(cur)}")
        else:
            self.music_title.configure(text=f"🎵 當前播放: {self.current_music_title}")

    def music_toggle(self):
        ok, msg = self.player.toggle()
        self.is_playing = self.player.is_playing
        self.btn_play.configure(text="暫停" if self.is_playing else "播放")
        self.update_music_title_from_current_file()
        self.update_vlc_status(msg)
        if not ok:
            messagebox.showinfo("音樂播放", msg)

    # ==============================
    # 設定更新
    # ==============================
    def open_settings_popup(self):
        CustomSettingsPopup(self)

    def update_contact_info(self, new_contact):
        self.contact_info = new_contact.strip()
        show_text = self.contact_info if self.contact_info else "尚未設定"
        self.contact_info_label.configure(text=f"緊急聯絡人: {show_text}")

    def update_music_title(self, new_music):
        self.current_music_title = new_music.strip() if new_music.strip() else self.current_music_title
        if self.player.current_file() is None:
            self.music_title.configure(text=f"🎵 當前播放: {self.current_music_title}")

    def update_breath_colors(self, colors):
        # demo 版固定配色，直接改檔案上方常數即可
        return

    # ==============================
    # Pain Index 與分級
    # ==============================
    
    def compute_pain_index(self, bpm, confidence):
        if bpm is None:
            return 0.0, 0.0, 0.0

        self.bpm_history.append(float(bpm))
        bpm_list = list(self.bpm_history)

        cur = bpm_list[-1]
        prev = bpm_list[-2] if len(bpm_list) >= 2 else cur
        slope = cur - prev

        mean = sum(bpm_list) / len(bpm_list)
        var = sum((x - mean) ** 2 for x in bpm_list) / len(bpm_list)
        variability = var ** 0.5

        # 14 BPM 以下幾乎不加分
        bpm_term = 5.0 * max(0.0, cur - 14.0)

        # 變化項保留，但不要太誇張
        slope_term = 1.5 * max(0.0, abs(slope) - 0.5)
        var_term = 1.0 * max(0.0, variability - 0.8)

        pain = bpm_term + slope_term + var_term

        # confidence 只做很小的縮放，不要再砍半
        conf_scale = 0.9 + 0.1 * max(0.0, min(1.0, confidence))
        pain *= conf_scale

        pain = max(0.0, min(100.0, pain))
        print(f"pain={pain}")
        return pain, slope, variability

    def get_pain_level(self, pain):
        if pain < self.level1_threshold:
            return 0
        elif pain < self.level2_threshold:
            return 1
        elif pain < self.level3_threshold:
            return 2
        return 3

    def pain_color(self, pain):
        if pain < self.level1_threshold:
            return PAIN_COLOR_L0
        elif pain < self.level2_threshold:
            return PAIN_COLOR_L1
        elif pain < self.level3_threshold:
            return PAIN_COLOR_L2
        return PAIN_COLOR_L3
    
    def stop_level2_guidance(self):
        if self.level2_loop_job is not None:
            try:
                self.after_cancel(self.level2_loop_job)
            except Exception:
                pass
            self.level2_loop_job = None

        if self.level2_intro_wait_job is not None:
            try:
                self.after_cancel(self.level2_intro_wait_job)
            except Exception:
                pass
            self.level2_intro_wait_job = None

        self.level2_intro_player.stop()
        self.inhale_player.stop()
        self.exhale_player.stop()
        self.level2_phase = 0

    def start_level2_guidance(self):
        self.stop_level2_guidance()
        self.level2_phase = 0

        if self.level2_intro_file and os.path.exists(self.level2_intro_file):
            ok, msg = self.level2_intro_player.play()
            self.update_vlc_status(msg if ok else f"Level 2 開場音檔未播放：{msg}")
            if ok:
                self.level2_intro_wait_job = self.after(200, self._wait_level2_intro_finish)
                return

        self._start_level2_breath_loop()

    def _wait_level2_intro_finish(self):
        if self.current_pain_level != 2:
            return

        if self.level2_intro_player.is_playing():
            self.level2_intro_wait_job = self.after(120, self._wait_level2_intro_finish)
        else:
            self.level2_intro_wait_job = None
            self._start_level2_breath_loop()

    def _start_level2_breath_loop(self):
        self._play_next_level2_cue()

    def _play_next_level2_cue(self):
        if self.current_pain_level != 2:
            return

        if self.level2_phase == 0:
            ok, msg = self.inhale_player.play()
            self.level2_phase = 1
        else:
            ok, msg = self.exhale_player.play()
            self.level2_phase = 0

        self.update_vlc_status(msg if ok else f"Level 2 語音未播放：{msg}")

        half_cycle_ms = int((60.0 / max(1.0, self.guided_bpm)) * 1000 / 2.0)

        # 讓短音檔播完後，再依 half cycle 繼續下一次提示
        self.level2_loop_job = self.after(max(400, half_cycle_ms), self._play_next_level2_cue)

    # ==============================
    # 回饋閉環
    # ==============================
    def apply_feedback_by_level(self, level):
        if level == self.last_feedback_level:
            return

        # 先全部清掉，再進入對應 level
        self.stop_level2_guidance()
        self.alert_player.stop()
        self.beep_alert.stop()
        self.alarm_active = False

        # 預設把音樂先停掉，只有 level 1 才恢復
        self.player.stop()
        self.is_playing = self.player.is_playing
        self.btn_play.configure(text="暫停" if self.is_playing else "播放")

        if level == 0:

            self.update_vlc_status("待機中")
            self.btn_play.configure(text="播放")

        elif level == 1:
            if self.enable_auto_music:
                ok, msg = self.player.auto_start()
                self.is_playing = self.player.is_playing
                self.btn_play.configure(text="暫停" if self.is_playing else "播放")
                self.update_music_title_from_current_file()
                self.update_vlc_status(msg)
            self.lbl_feedback_summary.configure(
                text="Level 1：已自動播放舒緩音樂，作為第一階段疼痛舒緩。"
            )

        elif level == 2:
            self.player.stop()
            self.is_playing = False
            self.btn_play.configure(text="播放")

            if self.enable_tts:
                self.start_level2_guidance()
            else:
                self.update_vlc_status("Level 2：語音提示已停用")

            self.lbl_feedback_summary.configure(
                text=f"Level 2：請跟著語音提示進行吸氣 / 吐氣（建議 {self.guided_bpm:.1f} BPM）。"
            )

        else:  # level == 3
            self.player.stop()
            self.stop_level2_guidance()
            self.is_playing = False
            self.btn_play.configure(text="播放")

            self.beep_alert.start()
            msg = self.beep_alert.last_status

            self.alarm_active = True
            self.update_vlc_status(msg)

            self.lbl_feedback_summary.configure(
                text="Level 3：高風險狀態，已停止其他音訊並啟動現場警報，請附近醫護立即留意。"
            )

        self.last_feedback_level = level

    # ==============================
    # 系統更新
    # ==============================
    def on_mode_change(self, mode):
        self.current_mode = mode
        self.save_settings_to_disk(show_popup=False)
        if mode == "校準模式":
            self.lbl_subtitle.configure(text="【校準中】請離開雷達前方進行背景噪音採樣...")
        else:
            self.lbl_subtitle.configure(text="系統監測中，請保持平穩呼吸...")

    def poll_engine(self):
        try:
            data = self.queue.get_nowait()
            self.update_breath_view(data)
        except queue.Empty:
            pass
        self.after(100, self.poll_engine)

    def update_breath_view(self, data):
        bpm = data.get("bpm", None)
        status = data.get("status", "UNCERTAIN")
        feedback_status = data.get("feedback_status", status)
        wave = data.get("wave", [])
        confidence = float(data.get("confidence", 0.0))

        self.current_bpm = bpm
        self.current_status = status
        self.current_confidence = confidence

        if bpm is None:
            self.val_bpm.configure(text="--", text_color="#9E9E9E")
            pain = 0.0
            slope = 0.0
            variability = 0.0
        else:
            self.val_bpm.configure(text=f"{bpm:.1f}", text_color=self._bpm_color(status))
            pain, slope, variability = self.compute_pain_index(bpm, confidence)

        self.current_slope = slope
        self.current_variability = variability


        if pain > self.display_pain:
            self.display_pain = min(self.display_pain + 2.0, pain)
        else:
            self.display_pain = max(self.display_pain - 1.5, pain)

        self.current_pain_index = self.display_pain
        self.current_pain_level = self.get_pain_level(self.current_pain_index)

        pain_color = self.pain_color(self.current_pain_index)
        self.lbl_level_big.configure(text=f"LEVEL {self.current_pain_level}", text_color=pain_color)

        self.lbl_status.configure(text=f"Status: {status}")

        if self.current_mode == "校準模式":
            self.lbl_subtitle.configure(text="【校準中】請離開雷達前方進行背景噪音採樣...")
        else:
            self.lbl_subtitle.configure(text=self._status_message(status, self.current_pain_level))

        self.apply_feedback_by_level(self.current_pain_level if feedback_status == "BREATHING" else 0)
        self.draw_waveform(wave)
    def draw_waveform(self, wave):
        self.wave_canvas.delete("all")
        w = max(self.wave_canvas.winfo_width(), 760)
        h = max(self.wave_canvas.winfo_height(), 95)
        self.wave_canvas.create_line(8, h / 2, w - 8, h / 2, fill="#3A3A3A", width=1)

        if not wave or len(wave) < 2:
            return

        arr = list(wave)
        vmin, vmax = min(arr), max(arr)
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-6

        pts = []
        for i, v in enumerate(arr):
            x = 10 + i * (w - 20) / max(1, len(arr) - 1)
            y = h - 10 - ((v - vmin) / (vmax - vmin)) * (h - 20)
            pts.extend([x, y])

        color = "#4DA3FF" if self.current_status == "BREATHING" else "#9E9E9E"
        self.wave_canvas.create_line(*pts, fill=color, width=2, smooth=True)

    def update_breathing_background(self):
        if self.current_mode == "校準模式":
            colors = [BREATH_LIGHT_IDLE, "#3A3333", BREATH_LIGHT_IDLE]
            interval = 900
        elif self.alarm_active and self.current_pain_level == 3:
            colors = ["#5A0000", "#2B2B2B", "#A00000", "#2B2B2B"]
            interval = 320
        elif self.current_pain_level >= 2:
            colors = [BREATH_LIGHT_LEVEL2_A]
            interval = 800
        elif self.current_pain_level == 1:
            colors = [BREATH_LIGHT_LEVEL1_A]
            interval = 1000
        elif self.current_status == "UNCERTAIN":
            colors = [BREATH_LIGHT_IDLE, "#31404D"]
            interval = 850
        else:
            colors = [BREATH_LIGHT_IDLE]
            interval = 1150

        self.breath_bg.configure(fg_color=colors[self.breath_step % len(colors)])
        self.breath_step = (self.breath_step + 1) % len(colors)
        self.after(interval, self.update_breathing_background)

    def _status_message(self, status, level):
        if status == "UNCERTAIN":
            return "系統正在確認呼吸狀態，請保持平穩並面向雷達。"
        if level == 0:
            return "系統監測中，Pain Index 尚未達回饋門檻。"
        if level == 1:
            return "Level 1：已啟動舒緩音樂，請保持平穩呼吸。"
        if level == 2:
            return f"Level 2：請跟著背景節奏進行吸氣 / 呼氣（建議 {self.guided_bpm:.1f} BPM）。"
        return "Level 3：高風險狀態，已啟動現場警報，請附近醫護立即留意。"

    def _bpm_color(self, status):
        if status == "BREATHING":
            return "#3498db"
        if status == "UNCERTAIN":
            return "#F1C40F"
        return "#9E9E9E"

    def on_close(self):
        self.save_settings_to_disk(show_popup=False)
        self.stop_event.set()
        self.player.stop()
        self.stop_level2_guidance()
        self.alert_player.stop()
        self.beep_alert.stop()
        self.destroy()


if __name__ == "__main__":
    app = BreathingMonitorApp()
    app.mainloop()
