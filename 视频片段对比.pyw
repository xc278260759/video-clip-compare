# -*- coding: utf-8 -*-
"""
视频片段对比 — 双轨 A/B 视频对比工具

- 两条独立时间轨 A / B，可左右拖动对齐（各自设置起点）
- 只在重叠时间段内播放与对比
- 四种对比模式：擦除（wipe）/ 叠加（overlay）/ 差分（difference）/ 并排（side by side）
- 中文界面，只保留对比所需功能

快捷键：空格 播放/暂停，←/→ 逐帧，Ctrl+O 打开 A，Ctrl+Shift+O 打开 B
"""

import os
import sys
import time
import ctypes
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

APP_NAME = "视频片段对比"
APP_ID = "XC.VideoClipCompare.Tool"
VIDEO_FILTER = [
    ("视频文件", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.mpg *.mpeg"),
    ("所有文件", "*.*"),
]

C_BG = "#1e1e1e"
C_BG2 = "#252525"
C_FG = "#e8e8e8"
C_DIM = "#9a9a9a"
C_LINE = "#2f2f2f"
C_ACCENT = "#e8423c"
C_TRACK_A = "#37618f"
C_TRACK_B = "#8a5a2b"
C_TRACK_OFF = "#464646"

FONT = ("Microsoft YaHei UI", 10)
FONT_S = ("Microsoft YaHei UI", 9)
FONT_B = ("Microsoft YaHei UI", 10, "bold")
FONT_TC = ("Consolas", 11)

MODES = [("wipe", "擦除"), ("overlay", "叠加"), ("diff", "差分"), ("side", "并排")]


def fmt_tc(t, fps):
    if t < 0:
        t = 0.0
    total = int(t)
    m, s = divmod(total, 60)
    f = int(round((t - total) * (fps or 25)))
    if f >= int(round(fps or 25)):
        f = 0
        s += 1
    return "%02d:%02d:%02d" % (m, s, f)


def fmt_short(t):
    t = max(0.0, t)
    m, s = divmod(int(round(t)), 60)
    return "%d:%02d" % (m, s)


class Clip(object):
    """一路视频：路径、帧率、总帧数、按时间取帧"""

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError("无法打开：%s" % path)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.duration = (self.total / self.fps) if self.total > 0 else 0.0
        self._last_idx = -1
        self._last_frame = None

    @property
    def name(self):
        return os.path.basename(self.path)

    def frame_at(self, t):
        """按片内时间取一帧（BGR）。越界返回 None。"""
        tol = 1.0 / max(1.0, self.fps)
        if t < -tol or (self.duration > 0 and t > self.duration + tol):
            return None
        if self.total <= 0:
            return None
        idx = int(round(t * self.fps))
        idx = max(0, min(idx, self.total - 1))
        if idx == self._last_idx and self._last_frame is not None:
            return self._last_frame
        if idx != self._last_idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        self._last_idx = idx
        self._last_frame = frame
        return frame

    def thumb(self, height=26):
        frame = self.frame_at(0)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        nw = max(1, int(w * (height / float(h))))
        nw = min(nw, 120)
        img = cv2.resize(frame, (nw, height), interpolation=cv2.INTER_AREA)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass


def fit_into(img, W, H):
    """等比缩放并居中到 W×H 黑底画布"""
    canvas = np.zeros((H, W, 3), np.uint8)
    if img is None:
        return canvas
    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0:
        return canvas
    scale = min(W / float(iw), H / float(ih))
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x = (W - nw) // 2
    y = (H - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def compose(a_img, b_img, mode, wipe, alpha, W, H):
    """把 A/B 两帧合成为一张画布"""
    if mode == "side":
        half = W // 2
        out = np.zeros((H, W, 3), np.uint8)
        out[:, :half] = fit_into(a_img, half, H)
        out[:, half:] = fit_into(b_img, W - half, H)
        cv2.line(out, (half, 0), (half, H - 1), (60, 60, 60), 1)
        return out

    A = fit_into(a_img, W, H)
    B = fit_into(b_img, W, H)

    if a_img is None and b_img is None:
        return A
    if a_img is None:
        return B
    if b_img is None:
        return A

    if mode == "overlay":
        return cv2.addWeighted(A, 1.0 - alpha, B, alpha, 0.0)

    if mode == "diff":
        d = cv2.absdiff(A, B)
        d = np.clip(d.astype(np.int16) * 3, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # wipe
    x = int(max(0, min(W, wipe * W)))
    out = A.copy()
    if x < W:
        out[:, x:] = B[:, x:]
    cv2.line(out, (x, 0), (x, H - 1), (60, 60, 232), 2)
    cv2.rectangle(out, (max(0, x - 8), H // 2 - 22), (min(W - 1, x + 8), H // 2 + 22), (60, 60, 232), -1)
    cv2.line(out, (x, H // 2 - 18), (x, H // 2 + 18), (255, 255, 255), 2)
    return out


class App(object):
    def __init__(self, root):
        self.root = root
        self.a = None
        self.b = None
        self.a_start = 0.0
        self.b_start = 0.0
        self.t = 0.0
        self.mode = "wipe"
        self.wipe = 0.5
        self.alpha = 0.5
        self.playing = False
        self.v0 = 0.0
        self.v1 = 10.0
        self.view_w = 960
        self.view_h = 540
        self.drag = None
        self.scrub_drag = False
        self.wipe_drag = False
        self.wipe_x = 0
        self.wipe_value = 0.0
        self.scrub_x = 0
        self.scrub_t = 0.0
        self.drag_x = 0
        self.drag_value = 0.0
        self.last_tick = time.time()
        self._tl_last = 0.0
        self.thumb_a = None
        self.thumb_b = None
        self.build_ui()
        self.root.after(15, self._tick)

    # ---------- 界面
    def build_ui(self):
        root = self.root
        root.title(APP_NAME)
        base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
        ico = os.path.join(base, "视频片段对比.ico")
        png = os.path.join(base, "视频片段对比.png")
        # 任务栏图标靠 iconphoto + AppUserModelID；标题栏靠 iconbitmap
        if os.path.exists(png):
            try:
                self._icon_img = tk.PhotoImage(file=png)
                root.iconphoto(True, self._icon_img)
            except Exception:
                pass
        if os.path.exists(ico):
            try:
                root.iconbitmap(ico)
            except Exception:
                pass
        root.configure(bg=C_BG)
        root.geometry("1280x860")
        root.minsize(900, 620)

        top = tk.Frame(root, bg=C_BG)
        top.pack(side="top", fill="x", padx=10, pady=(8, 6))

        def mk_btn(text, cmd, width=None):
            b = tk.Button(top, text=text, command=cmd, bg=C_BG2, fg=C_FG, activebackground="#333333",
                          activeforeground=C_FG, relief="flat", bd=0, font=FONT_S, cursor="hand2",
                          padx=10, pady=3, width=width)
            return b

        mk_btn("打开 A", self.open_a).pack(side="left", padx=(0, 4))
        mk_btn("打开 B", self.open_b).pack(side="left", padx=(0, 14))

        tk.Label(top, text="模式", bg=C_BG, fg=C_DIM, font=FONT_S).pack(side="left", padx=(0, 4))
        self.mode_var = tk.StringVar(value="擦除")
        self.mode_menu = tk.OptionMenu(top, self.mode_var, *[m[1] for m in MODES], command=self.on_mode)
        self.mode_menu.configure(bg=C_BG2, fg=C_FG, activebackground="#333333", activeforeground=C_FG,
                                 relief="flat", bd=0, highlightthickness=0, font=FONT_S, width=6)
        self.mode_menu.pack(side="left", padx=(0, 12))

        tk.Label(top, text="透明度", bg=C_BG, fg=C_DIM, font=FONT_S).pack(side="left", padx=(0, 2))
        self.alpha_var = tk.DoubleVar(value=50)
        self.alpha_scale = tk.Scale(top, from_=0, to=100, orient="horizontal", variable=self.alpha_var,
                                    command=self.on_alpha, bg=C_BG, fg=C_FG, troughcolor="#333333",
                                    highlightthickness=0, bd=0, length=90, font=FONT_S, showvalue=0)
        self.alpha_scale.pack(side="left", padx=(0, 14))

        self.play_btn = mk_btn("▶ 播放", self.toggle_play)
        self.play_btn.configure(width=8)
        self.play_btn.pack(side="left", padx=(0, 6))
        mk_btn("◀ 帧", lambda: self.step(-1)).pack(side="left", padx=(0, 4))
        mk_btn("帧 ▶", lambda: self.step(1)).pack(side="left", padx=(0, 14))

        self.tc_label = tk.Label(top, text="00:00:00", bg=C_BG, fg=C_FG, font=FONT_TC)
        self.tc_label.pack(side="left")

        # 状态栏在最底，时间轴压在其上，画面占剩下的全部空间
        self.status = tk.Label(root, text="打开两个视频开始对比", bg=C_BG, fg=C_DIM, font=FONT_S, anchor="w")
        self.status.pack(side="bottom", fill="x", padx=12, pady=(2, 6))

        self.tl = tk.Canvas(root, bg=C_BG2, height=self.TL_H, highlightthickness=0, bd=0)
        self.tl.pack(side="bottom", fill="x", padx=8, pady=(0, 4))
        self.tl.bind("<Configure>", lambda e: self.draw_timeline())
        self.tl.bind("<Button-1>", self.tl_press)
        self.tl.bind("<B1-Motion>", self.tl_move)
        self.tl.bind("<ButtonRelease-1>", self.tl_release)
        self.tl.bind("<MouseWheel>", self.tl_wheel)

        self.screen = tk.Canvas(root, bg="#000000", highlightthickness=0, bd=0)
        self.screen.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 6))
        self.screen.bind("<Configure>", self.on_screen_resize)
        self.screen.bind("<Button-1>", self.on_screen_press)
        self.screen.bind("<B1-Motion>", self.on_screen_drag)
        self.screen.bind("<Motion>", self.on_screen_motion)
        self.screen.bind("<ButtonRelease-1>", self.on_screen_release)

        root.bind("<space>", lambda e: self.toggle_play())
        root.bind("<Left>", lambda e: self.step(-1))
        root.bind("<Right>", lambda e: self.step(1))
        root.bind("<Control-o>", lambda e: self.open_a())
        root.bind("<Control-O>", lambda e: self.open_b())
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- 打开 / 状态
    def open_a(self):
        p = filedialog.askopenfilename(title="选择 A 视频", filetypes=VIDEO_FILTER)
        if p:
            self.load("a", p)

    def open_b(self):
        p = filedialog.askopenfilename(title="选择 B 视频", filetypes=VIDEO_FILTER)
        if p:
            self.load("b", p)

    def load(self, side, path):
        try:
            clip = Clip(path)
        except Exception as e:
            messagebox.showerror(APP_NAME, "打开失败：\n%s" % e)
            return
        old = self.a if side == "a" else self.b
        if old:
            old.close()
        if side == "a":
            self.a = clip
            self.a_start = 0.0
            self.thumb_a = self._make_thumb(clip)
        else:
            self.b = clip
            # B 默认与 A 起点对齐
            self.b_start = self.a_start if self.a else 0.0
            self.thumb_b = self._make_thumb(clip)
        self.fit_view()
        ov = self.play_range()
        if ov:
            self.t = ov[0]
        self.playing = False
        self.play_btn.configure(text="▶ 播放")
        self.refresh_status()
        self.render()
        self.draw_timeline()

    def _make_thumb(self, clip):
        im = clip.thumb(26)
        if im is None:
            return None
        return ImageTk.PhotoImage(im)

    def refresh_status(self):
        parts = []
        if self.a:
            parts.append("A %s  %.2fs" % (self.a.name, self.a.duration))
        if self.b:
            parts.append("B %s  %.2fs" % (self.b.name, self.b.duration))
        ov = self.overlap()
        if ov:
            parts.append("重叠 %.2fs" % (ov[1] - ov[0]))
        elif self.a and self.b:
            parts.append("无重叠")
        if self.a and self.b:
            parts.append("B 偏移 %+.2fs" % (self.b_start - self.a_start))
        if self.a or self.b:
            parts.append("拖画面扫帧 · 抓红色把手移擦除线")
        self.status.configure(text="   ·   ".join(parts), fg=(C_ACCENT if (self.a and self.b and not ov) else C_DIM))

    # ---------- 时间轴范围
    def fit_view(self):
        starts = []
        ends = []
        if self.a:
            starts.append(self.a_start)
            ends.append(self.a_start + self.a.duration)
        if self.b:
            starts.append(self.b_start)
            ends.append(self.b_start + self.b.duration)
        if not starts:
            self.v0, self.v1 = 0.0, 10.0
            return
        t0 = min(starts)
        t1 = max(ends)
        span = max(0.1, t1 - t0)
        self.v0 = t0 - span * 0.05
        self.v1 = t1 + span * 0.05

    def overlap(self):
        if not (self.a and self.b):
            return None
        a0, a1 = self.a_start, self.a_start + self.a.duration
        b0, b1 = self.b_start, self.b_start + self.b.duration
        o0, o1 = max(a0, b0), min(a1, b1)
        if o1 <= o0 + 1e-6:
            return None
        return (o0, o1)

    def play_range(self):
        ov = self.overlap()
        if ov:
            return ov
        if self.a:
            return (self.a_start, self.a_start + self.a.duration)
        if self.b:
            return (self.b_start, self.b_start + self.b.duration)
        return None

    # ---------- 渲染
    def on_screen_resize(self, event):
        if event.width > 10 and event.height > 10:
            self.view_w, self.view_h = event.width, event.height
            self.render()

    def render(self):
        if self.view_w < 2 or self.view_h < 2:
            return
        a_img = self.a.frame_at(self.t - self.a_start) if self.a else None
        b_img = self.b.frame_at(self.t - self.b_start) if self.b else None
        img = compose(a_img, b_img, self.mode, self.wipe, self.alpha, self.view_w, self.view_h)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(pil)
        c = self.screen
        if c.find_withtag("img"):
            c.itemconfigure("img", image=self.photo)
        else:
            c.create_image(0, 0, image=self.photo, anchor="nw", tags="img")
        fps = self.a.fps if self.a else (self.b.fps if self.b else 25)
        self.tc_label.configure(text=fmt_tc(self.t, fps))

    def draw_scrub(self):
        pass


    # 时间轴内部纵向布局：刻度行在最上，下面依次 A 轨、B 轨（不留多余空白）
    TL_SCALE_H = 24
    TL_A0, TL_A1 = 26, 56
    TL_B0, TL_B1 = 58, 88
    TL_H = 90

    def draw_timeline(self):
        c = self.tl
        c.delete("all")
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 20:
            return
        v0, v1 = self.v0, self.v1
        span = max(1e-6, v1 - v0)
        A0, A1, B0, B1 = self.TL_A0, self.TL_A1, self.TL_B0, self.TL_B1
        scale_h = self.TL_SCALE_H

        def X(t):
            return (t - v0) / span * W

        # 轨道底色（非重叠段用灰）
        for side, y0, y1 in (("a", A0, A1), ("b", B0, B1)):
            clip = self.a if side == "a" else self.b
            if not clip:
                c.create_rectangle(0, y0, W, y1, fill="#2a2a2a", outline="")
                c.create_text(8, (y0 + y1) // 2, text=("A 轨" if side == "a" else "B 轨") + "（未载入）",
                              anchor="w", fill=C_DIM, font=FONT_S)
                continue
            start = self.a_start if side == "a" else self.b_start
            x0, x1 = X(start), X(start + clip.duration)
            color = C_TRACK_A if side == "a" else C_TRACK_B
            ov = self.overlap()
            if ov:
                ox0, ox1 = max(x0, X(ov[0])), min(x1, X(ov[1]))
                if ox1 > ox0:
                    c.create_rectangle(x0, y0, ox0, y1, fill=C_TRACK_OFF, outline="")
                    c.create_rectangle(ox1, y0, x1, y1, fill=C_TRACK_OFF, outline="")
                    c.create_rectangle(ox0, y0, ox1, y1, fill=color, outline="")
                else:
                    c.create_rectangle(x0, y0, x1, y1, fill=C_TRACK_OFF, outline="")
            else:
                c.create_rectangle(x0, y0, x1, y1, fill=C_TRACK_OFF, outline="")
            label = ("A  " if side == "a" else "B  ") + clip.name
            c.create_text(max(6, x0 + 4), (y0 + y1) // 2, text=label, anchor="w", fill="#ffffff", font=FONT_S)
            thumb = self.thumb_a if side == "a" else self.thumb_b
            if thumb is not None:
                c.create_image(max(2, x0 + 2), y0 + 2, image=thumb, anchor="nw")

        # 刻度行底
        c.create_rectangle(0, 0, W, scale_h, fill="#2b2b2b", outline="")
        c.create_line(0, scale_h, W, scale_h, fill="#3a3a3a")

        # 刻度与竖向网格
        target = span / 8.0
        steps = [0.04, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]
        step = steps[-1]
        for s in steps:
            if s >= target:
                step = s
                break
        t = int(v0 / step) * step
        while t <= v1:
            if t >= v0:
                x = X(t)
                c.create_line(x, scale_h, x, B1, fill="#3a3a3a")
                c.create_line(x, scale_h - 4, x, scale_h + 4, fill="#6a6a6a")
                c.create_text(x + 3, 4, text=fmt_short(t), anchor="nw", fill=C_DIM, font=FONT_S)
            t += step

        # 重叠区间提示（只用绿框和顶部绿条，不再占一行空白）
        ov = self.overlap()
        if ov and self.a and self.b:
            ox0, ox1 = X(ov[0]), X(ov[1])
            c.create_rectangle(ox0, A0 - 1, ox1, B1 + 1, outline="#4a7a4a", width=1)
            c.create_line(ox0, A0, ox1, A0, fill="#6fcf6f", width=3)

        # 播放头
        if self.v0 <= self.t <= self.v1:
            x = X(self.t)
            c.create_line(x, 0, x, B1, fill=C_ACCENT, width=2)
            c.create_polygon(x - 5, 0, x + 5, 0, x, 7, fill=C_ACCENT, outline="")

    # ---------- 时间轴交互
    def tl_press(self, e):
        W = self.tl.winfo_width() or 1
        if self.TL_A0 <= e.y <= self.TL_A1 and self.a:
            self.drag = "a"
            self.drag_value = self.a_start
        elif self.TL_B0 <= e.y <= self.TL_B1 and self.b:
            self.drag = "b"
            self.drag_value = self.b_start
        else:
            self.drag = "seek"
            self.seek_px(e.x, W)
        self.drag_x = e.x
        self.draw_timeline()

    def tl_move(self, e):
        W = self.tl.winfo_width() or 1
        span = self.v1 - self.v0
        if self.drag in ("a", "b"):
            dt = (e.x - self.drag_x) / float(W) * span
            if self.drag == "a":
                self.a_start = self.drag_value + dt
            else:
                self.b_start = self.drag_value + dt
            ov = self.play_range()
            if ov:
                self.t = max(ov[0], min(self.t, ov[1]))
            self.refresh_status()
            self.render()
            self.draw_timeline()
        elif self.drag == "seek":
            self.seek_px(e.x, W)

    def tl_release(self, e):
        self.drag = None

    def seek_px(self, x, W):
        t = self.v0 + (x / float(W)) * (self.v1 - self.v0)
        r = self.play_range()
        if r:
            t = max(r[0], min(t, r[1]))
        self.t = t
        self.render()
        self.draw_timeline()

    def tl_wheel(self, e):
        W = self.tl.winfo_width() or 1
        anchor = self.v0 + (e.x / float(W)) * (self.v1 - self.v0)
        span = (self.v1 - self.v0) * (0.82 if e.delta > 0 else 1.22)
        span = max(0.2, min(span, 7200))
        self.v0 = anchor - (e.x / float(W)) * span
        self.v1 = self.v0 + span
        self.draw_timeline()

    # ---------- 画面交互
    # 拖画面任意处 = 相对扫帧（抓片手感）；抓红色把手（或 Alt+靠近红线）= 移动擦除线
    def wipe_handle_hit(self, x, y):
        if self.mode != "wipe" or self.view_w <= 0:
            return False
        cx = self.wipe * self.view_w
        cy = self.view_h // 2
        return abs(x - cx) <= 12 and abs(y - cy) <= 30

    def on_screen_press(self, e):
        alt_down = bool(e.state & 0x0008)
        near_line = (self.mode == "wipe" and self.view_w > 0
                     and abs(e.x - self.wipe * self.view_w) <= 12)
        if self.wipe_handle_hit(e.x, e.y) or (alt_down and near_line):
            self.wipe_drag = True
            self.wipe_x = e.x
            self.wipe_value = self.wipe
            self.scrub_drag = False
            self.screen.configure(cursor="sb_h_double_arrow")
            return
        self.wipe_drag = False
        self.scrub_drag = True
        self.scrub_x = e.x
        self.scrub_t = self.t
        if self.playing:
            self.playing = False
            self.play_btn.configure(text="▶ 播放")

    def on_screen_motion(self, e):
        if self.wipe_drag or self.scrub_drag:
            return
        alt_down = bool(e.state & 0x0008)
        near_line = (self.mode == "wipe" and self.view_w > 0
                     and abs(e.x - self.wipe * self.view_w) <= 12)
        self.screen.configure(cursor="sb_h_double_arrow"
                              if (self.wipe_handle_hit(e.x, e.y) or (alt_down and near_line)) else "")

    def on_screen_drag(self, e):
        if self.wipe_drag and self.view_w > 0:
            self.wipe = max(0.0, min(1.0, self.wipe_value + (e.x - self.wipe_x) / float(self.view_w)))
            self.render()
        elif self.scrub_drag:
            self.scrub_to(e.x)

    def on_screen_release(self, e):
        self.scrub_drag = False
        self.wipe_drag = False
        self.screen.configure(cursor="")

    def scrub_to(self, x):
        """相对拖动：相对按下点位移多少，时间就走多少"""
        r = self.play_range()
        if not r:
            return
        W = max(1, self.view_w)
        t0, t1 = r
        dt = (x - self.scrub_x) / float(W) * (t1 - t0)
        t = self.scrub_t + dt
        fps = self.a.fps if self.a else (self.b.fps if self.b else 25)
        t = round(t * fps) / fps
        self.t = max(t0, min(t, t1))
        self.render()
        self.draw_timeline()

    # ---------- 播放
    def toggle_play(self):
        if not (self.a or self.b):
            return
        if self.playing:
            self.playing = False
            self.play_btn.configure(text="▶ 播放")
        else:
            r = self.play_range()
            if r and (self.t < r[0] or self.t >= r[1]):
                self.t = r[0]
            self.playing = True
            self.last_tick = time.time()
            self.play_btn.configure(text="⏸ 暂停")

    def step(self, n):
        fps = self.a.fps if self.a else (self.b.fps if self.b else 25)
        r = self.play_range()
        self.t += n / float(fps)
        if r:
            if self.t > r[1]:
                self.t = r[0] if n > 0 else r[1]
            if self.t < r[0]:
                self.t = r[1] if n < 0 else r[0]
        self.render()
        self.draw_timeline()

    def _tick(self):
        if self.playing:
            now = time.time()
            dt = now - self.last_tick
            self.last_tick = now
            r = self.play_range()
            if r:
                self.t += dt
                if self.t >= r[1]:
                    self.t = r[0]
            self.render()
            if now - self._tl_last > 0.12:
                self._tl_last = now
                self.draw_timeline()
        self.root.after(15, self._tick)

    def on_mode(self, label):
        for key, name in MODES:
            if name == label:
                self.mode = key
                break
        self.render()

    def on_alpha(self, value):
        self.alpha = float(value) / 100.0
        if self.mode == "overlay":
            self.render()

    def on_close(self):
        self.playing = False
        if self.a:
            self.a.close()
        if self.b:
            self.b.close()
        self.root.destroy()


def main():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
