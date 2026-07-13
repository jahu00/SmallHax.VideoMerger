#!/usr/bin/env python3
"""
Dual-video frame-accurate comparison tool.

Goal: open two videos side by side, play / pause / stop / seek / step frames,
mark two frames in each video, and compute the linear mapping y = a*x + b
between the two timelines (in seconds). Those coefficients will later be used
to time-stretch audio from one video to match the other via system ffmpeg.

Design notes:
  * No third-party pip modules. Frames are decoded by the system `ffmpeg`
    (raw rgb24 over a pipe) and shown with tkinter.PhotoImage fed a PPM buffer.
  * Metadata comes from `ffprobe`.
  * Each player owns a persistent ffmpeg process producing sequential frames,
    plus a background reader thread filling a bounded queue. The main (UI)
    thread is the only place that touches tkinter.
  * Frame indices are tracked deterministically: after seeking to frame N via
    input `-ss (N/fps)`, the first delivered frame is treated as frame N and
    the counter increments per frame after that.
"""

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Optional drag-and-drop support via the tkdnd Tcl extension (tkinterdnd2).
# The app runs fine without it; drops are simply disabled in that case.
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _DND_AVAILABLE = True
except Exception:  # noqa: BLE001
    TkinterDnD = None
    DND_FILES = None
    _DND_AVAILABLE = False


# ----------------------------------------------------------------------------
# ffmpeg / ffprobe helpers
# ----------------------------------------------------------------------------
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

# Max width used for on-screen display. Decoding at a reduced size keeps the
# pipe cheap and playback smooth. It does not affect frame indexing/timing.
DISPLAY_MAX_W = 480


def _parse_rate(text):
    """Parse an ffprobe rate string like '30000/1001' -> float fps."""
    if not text or text in ("0/0", "N/A"):
        return 0.0
    if "/" in text:
        num, den = text.split("/")
        den = float(den)
        return float(num) / den if den else 0.0
    return float(text)


def probe(path):
    """Return metadata dict for the first video stream."""
    cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    data = json.loads(out)
    stream = data["streams"][0]
    fmt = data.get("format", {})

    fps = _parse_rate(stream.get("avg_frame_rate"))
    if fps <= 0:
        fps = _parse_rate(stream.get("r_frame_rate"))
    if fps <= 0:
        fps = 25.0  # last-resort fallback

    duration = 0.0
    for src in (stream.get("duration"), fmt.get("duration")):
        try:
            duration = float(src)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    try:
        nb_frames = int(stream.get("nb_frames"))
    except (TypeError, ValueError):
        nb_frames = 0
    if nb_frames <= 0 and duration > 0:
        nb_frames = int(round(duration * fps))

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration": duration,
        "nb_frames": max(nb_frames, 1),
        "audio_tracks": probe_audio_tracks(path),
    }


def probe_audio_tracks(path):
    """Return a list of audio stream descriptors for the file.

    Each entry: {"index": <stream index within audio streams, 0-based>,
                 "label": <human-readable description>}.
    The index is relative to audio streams (i.e. maps to ffmpeg's a:N),
    which is what we'll want when selecting a track for playback/muxing later.
    """
    cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", "a",
        "-show_entries",
        "stream=index,codec_name,channels,sample_rate:stream_tags=language,title",
        "-of", "json", path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        streams = json.loads(out).get("streams", [])
    except Exception:  # noqa: BLE001
        return []

    tracks = []
    for a_idx, s in enumerate(streams):
        tags = s.get("tags", {}) or {}
        parts = [f"#{a_idx}"]
        title = tags.get("title")
        if title:
            parts.append(title)
        lang = tags.get("language")
        if lang:
            parts.append(f"[{lang}]")
        codec = s.get("codec_name")
        if codec:
            parts.append(codec)
        ch = s.get("channels")
        if ch:
            parts.append({1: "mono", 2: "stereo"}.get(ch, f"{ch}ch"))
        sr = s.get("sample_rate")
        if sr:
            parts.append(f"{sr}Hz")
        tracks.append({"index": a_idx, "label": " ".join(parts)})
    return tracks


def probe_audio_sample_rate(path, audio_index):
    """Return the sample rate (int) of audio stream a:audio_index, or 48000."""
    cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", f"a:{audio_index}",
        "-show_entries", "stream=sample_rate",
        "-of", "json", path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        streams = json.loads(out).get("streams", [])
        if streams:
            return int(streams[0]["sample_rate"])
    except Exception:  # noqa: BLE001
        pass
    return 48000


def probe_duration(path):
    """Return container duration in seconds, or 0.0 if unknown."""
    cmd = [FFPROBE, "-v", "error", "-show_entries", "format=duration",
           "-of", "json", path]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return float(json.loads(out)["format"]["duration"])
    except Exception:  # noqa: BLE001
        return 0.0


def atempo_chain(tempo):
    """Decompose a tempo factor into a chain of atempo filters.

    A single atempo only accepts factors in [0.5, 2.0]; larger/smaller ratios
    are achieved by chaining. Returns a list of filter strings.
    """
    filters = []
    t = float(tempo)
    if t <= 0:
        t = 1.0
    while t > 2.0:
        filters.append("atempo=2.0")
        t /= 2.0
    while t < 0.5:
        filters.append("atempo=0.5")
        t *= 2.0
    filters.append(f"atempo={t:.6f}")
    return filters


def fit_size(src_w, src_h, box_w, box_h):
    """Largest even-dimensioned size fitting box_w x box_h, keeping aspect.

    Never upscales beyond the source resolution (decoding larger than the
    source just wastes work and looks no sharper).
    """
    if src_w <= 0 or src_h <= 0:
        return 2, 2
    box_w = max(box_w, 2)
    box_h = max(box_h, 2)
    scale = min(box_w / src_w, box_h / src_h, 1.0)
    dw = int(src_w * scale)
    dh = int(src_h * scale)
    dw -= dw % 2
    dh -= dh % 2
    return max(dw, 2), max(dh, 2)


# ----------------------------------------------------------------------------
# Audio sync exporter
# ----------------------------------------------------------------------------
class AudioExportJob:
    """Runs the two-pass audio sync export in a background thread.

    The linear timeline mapping is  t_B = a * t_A + b  (seconds).

    To make source audio play in sync with the *target* video timeline we:
      * stretch the source by the ratio that maps its own time onto the
        target time, then
      * offset it, then
      * trim/pad it to exactly the target video's duration so the audio never
        extends beyond the target video.

    direction="A_to_B": take A's audio, fit it onto B's timeline.
        stretch = a ,  offset = b ,  target duration = B's video duration.
    direction="B_to_A": take B's audio, fit it onto A's timeline.
        Inverting the mapping gives t_A = (1/a) * t_B + (-b/a).
        stretch = 1/a , offset = -b/a , target duration = A's video duration.

    Two passes are used deliberately: combining tempo change and delay/trim in
    a single filtergraph produced incorrect timestamps, so pass 1 handles
    tempo/pitch (+ optional loudnorm) and pass 2 handles offset + pad/trim.
    """

    def __init__(self, src_path, src_audio_index, stretch, offset,
                 target_duration, out_path, *, preserve_pitch=True,
                 normalize=False, codec="aac", bitrate="320k", channels=2,
                 log_cb=None, done_cb=None):
        self.src_path = src_path
        self.src_audio_index = src_audio_index
        self.stretch = stretch
        self.offset = offset
        self.target_duration = target_duration
        self.out_path = out_path
        self.preserve_pitch = preserve_pitch
        self.normalize = normalize
        self.codec = codec
        self.bitrate = bitrate
        self.channels = channels
        self.log_cb = log_cb or (lambda line: None)
        self.done_cb = done_cb or (lambda ok, msg: None)

        self._proc = None
        self._cancelled = threading.Event()
        self._thread = None
        self._tmpfiles = []

    # --- public API --------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancelled.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass

    # --- filter construction ----------------------------------------------
    def _pass1_filter(self, sr):
        f = []
        if self.preserve_pitch:
            f += atempo_chain(1.0 / self.stretch)
        else:
            # asetrate changes pitch with speed; resample back to sr afterwards
            f.append(f"asetrate={sr / self.stretch:.6f}")
            f.append(f"aresample={sr}")
        if self.normalize:
            f.append("loudnorm=I=-16:TP=-1.5:LRA=11")
        f.append(f"aresample={sr}")
        return ",".join(f)

    def _pass2_filter(self):
        f = []
        ms = int(round(self.offset * 1000))
        if ms > 0:
            f.append(f"adelay={ms}:all=1")
        elif ms < 0:
            # negative offset: source starts before the target -> trim front
            f.append(f"atrim=start={-self.offset:.6f}")
            f.append("asetpts=PTS-STARTPTS")
        # cap to target length (never extend beyond target video), then pad
        f.append(f"atrim=end={self.target_duration:.6f}")
        f.append("asetpts=PTS-STARTPTS")
        f.append(f"apad=whole_dur={self.target_duration:.6f}")
        return ",".join(f)

    # --- execution ---------------------------------------------------------
    def _run(self):
        try:
            if self.target_duration <= 0:
                raise ValueError("Target video duration is unknown or zero.")
            sr = probe_audio_sample_rate(self.src_path, self.src_audio_index)
            self.log_cb(f"# source sample rate: {sr} Hz\n")
            self.log_cb(f"# stretch factor: {self.stretch:.6f}   "
                        f"offset: {self.offset:+.6f} s   "
                        f"target: {self.target_duration:.3f} s\n")

            fd, mid = tempfile.mkstemp(suffix=".wav", prefix="vsync_mid_")
            os.close(fd)
            self._tmpfiles.append(mid)

            # Pass 1: tempo/pitch (+normalize) into a lossless intermediate.
            cmd1 = [
                FFMPEG, "-hide_banner", "-y",
                "-i", self.src_path,
                "-map", f"0:a:{self.src_audio_index}",
                "-af", self._pass1_filter(sr),
                "-c:a", "pcm_s16le",
                mid,
            ]
            if not self._run_ffmpeg(cmd1, "PASS 1/2  (tempo / pitch"
                                          + (" + normalize" if self.normalize
                                             else "") + ")"):
                return

            # Pass 2: offset + pad/trim, encode to the chosen codec.
            cmd2 = [
                FFMPEG, "-hide_banner", "-y",
                "-i", mid,
                "-af", self._pass2_filter(),
                "-c:a", self.codec,
                "-ac", str(self.channels),
            ]
            if self.codec != "flac":  # flac ignores bitrate
                cmd2 += ["-b:a", self.bitrate]
            cmd2 += [self.out_path]
            if not self._run_ffmpeg(cmd2, "PASS 2/2  (offset + pad/trim + encode)"):
                return

            self._cleanup_tmp()
            self.done_cb(True, f"Saved: {self.out_path}")
        except Exception as exc:  # noqa: BLE001
            self._cleanup_tmp()
            self.log_cb(f"\n[error] {exc}\n")
            self.done_cb(False, str(exc))

    def _run_ffmpeg(self, cmd, title):
        if self._cancelled.is_set():
            self.done_cb(False, "Cancelled")
            return False
        self.log_cb(f"\n===== {title} =====\n")
        self.log_cb("$ " + " ".join(_shell_quote(c) for c in cmd) + "\n\n")
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, bufsize=1,
        )
        for line in self._proc.stderr:
            self.log_cb(line)
        self._proc.wait()
        rc = self._proc.returncode
        self._proc = None
        if self._cancelled.is_set():
            self.done_cb(False, "Cancelled")
            return False
        if rc != 0:
            self.log_cb(f"\n[ffmpeg exited with code {rc}]\n")
            self.done_cb(False, f"ffmpeg failed (exit {rc})")
            return False
        return True

    def _cleanup_tmp(self):
        for p in self._tmpfiles:
            try:
                os.remove(p)
            except OSError:
                pass
        self._tmpfiles = []


def _shell_quote(s):
    if not s or any(c in s for c in ' \t"\'\\'):
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return s


# ----------------------------------------------------------------------------
# VideoPlayer: owns one ffmpeg decode pipe + reader thread
# ----------------------------------------------------------------------------
class VideoPlayer:
    def __init__(self):
        self.path = None
        self.meta = None
        self.fps = 0.0
        self.total_frames = 0
        self.src_w = self.src_h = 0   # native video resolution
        self.dw = self.dh = 0         # current decode/display resolution
        self.box_w = self.box_h = 0   # target display area (from the UI)
        self.frame_bytes = 0

        self._proc = None
        self._thread = None
        self._queue = queue.Queue(maxsize=8)
        self._gen = 0                 # generation id; bumps on every (re)spawn
        self._lock = threading.Lock()

        self.current_index = 0        # index of the frame last displayed
        self.playing = False
        self.audio_tracks = []        # list from probe_audio_tracks
        self.audio_index = None       # selected audio stream (a:N) or None

    # --- lifecycle ---------------------------------------------------------
    def open(self, path):
        self._stop_current()
        self.path = path
        self.meta = probe(path)
        self.fps = self.meta["fps"]
        self.total_frames = self.meta["nb_frames"]
        self.src_w = self.meta["width"]
        self.src_h = self.meta["height"]
        self.audio_tracks = self.meta.get("audio_tracks", [])
        self.audio_index = self.audio_tracks[0]["index"] if self.audio_tracks else None
        self._recompute_display()
        self.current_index = 0
        self.playing = False
        self._spawn(0)

    def set_display_box(self, box_w, box_h):
        """Record the available display area; return True if decode size changed."""
        if box_w <= 0 or box_h <= 0:
            return False
        self.box_w, self.box_h = box_w, box_h
        if self.path is None:
            return False
        old = (self.dw, self.dh)
        self._recompute_display()
        return (self.dw, self.dh) != old

    def _recompute_display(self):
        if self.box_w > 0 and self.box_h > 0:
            self.dw, self.dh = fit_size(self.src_w, self.src_h,
                                        self.box_w, self.box_h)
        else:
            # no box known yet: fall back to native, capped for safety
            self.dw, self.dh = fit_size(self.src_w, self.src_h,
                                        DISPLAY_MAX_W, 10 ** 9)
        self.frame_bytes = self.dw * self.dh * 3

    def rescale(self):
        """Re-decode from the current frame at the new display size."""
        if self.path is None:
            return
        self.seek(self.current_index)

    def close(self):
        self._stop_current()

    def _stop_current(self):
        """Invalidate the current generation, kill the process, drain queue."""
        with self._lock:
            self._gen += 1
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.stdout.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
        # drain
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def _spawn(self, start_frame):
        """Start an ffmpeg pipe delivering sequential frames from start_frame."""
        start_frame = max(0, min(start_frame, self.total_frames - 1))
        with self._lock:
            self._gen += 1
            gen = self._gen

        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
        if start_frame > 0:
            t = start_frame / self.fps
            cmd += ["-ss", f"{t:.6f}"]
        cmd += [
            "-i", self.path,
            "-vf", f"scale={self.dw}:{self.dh}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, bufsize=self.frame_bytes,
            close_fds=True,
        )
        self._proc = proc
        self._thread = threading.Thread(
            target=self._reader, args=(proc, gen, start_frame), daemon=True
        )
        self._thread.start()

    def _reader(self, proc, gen, start_frame):
        idx = start_frame
        stdout = proc.stdout
        n = self.frame_bytes
        while True:
            with self._lock:
                if gen != self._gen:
                    break
            data = stdout.read(n)
            if not data or len(data) < n:
                break  # EOF or torn read (likely killed)
            # Block until space; but bail out if this generation is stale.
            while True:
                with self._lock:
                    if gen != self._gen:
                        return
                try:
                    self._queue.put((gen, idx, data), timeout=0.1)
                    break
                except queue.Full:
                    continue
            idx += 1

    # --- frame access ------------------------------------------------------
    def get_next(self, block_timeout=0.0):
        """Return (index, ppm_bytes) for the next sequential frame or None."""
        deadline = block_timeout
        while True:
            try:
                gen, idx, data = self._queue.get(timeout=0.05 if deadline else 0)
            except queue.Empty:
                if deadline:
                    deadline -= 0.05
                    if deadline > 0:
                        continue
                return None
            with self._lock:
                stale = gen != self._gen
            if stale:
                continue
            self.current_index = idx
            return idx, self._to_ppm(data)

    def _to_ppm(self, rgb):
        header = b"P6\n%d %d\n255\n" % (self.dw, self.dh)
        return header + rgb

    # --- controls ----------------------------------------------------------
    def seek(self, frame):
        frame = max(0, min(frame, self.total_frames - 1))
        self._stop_current()
        self._spawn(frame)
        # current_index will be corrected once the first frame is displayed;
        # set an optimistic value so labels are sane if display lags.
        self.current_index = frame

    def time_at(self, frame):
        return frame / self.fps if self.fps else 0.0


# ----------------------------------------------------------------------------
# UI panel for a single video
# ----------------------------------------------------------------------------
class VideoPanel(ttk.Frame):
    def __init__(self, master, app, title):
        super().__init__(master, padding=6)
        self.app = app
        self.player = VideoPlayer()
        self._imgref = None           # keep PhotoImage alive
        self.mark1 = None             # frame index of mark 1
        self.mark2 = None

        ttk.Label(self, text=title, font=("TkDefaultFont", 10, "bold")).pack()

        self.name_var = tk.StringVar(value="(no file)")
        ttk.Label(self, textvariable=self.name_var, foreground="#555").pack()

        # audio track selector (no playback yet; used later for muxing)
        audio_row = ttk.Frame(self)
        audio_row.pack(pady=2)
        ttk.Label(audio_row, text="Audio:").pack(side="left", padx=(0, 4))
        self.audio_var = tk.StringVar(value="(none)")
        self.audio_combo = ttk.Combobox(
            audio_row, textvariable=self.audio_var, state="disabled",
            width=32, values=[],
        )
        self.audio_combo.pack(side="left")
        self.audio_combo.bind("<<ComboboxSelected>>", self._on_audio_selected)

        # Container that expands to fill available space; the video Label is
        # centered inside it. We measure this container to size the decode.
        self.video_area = tk.Frame(self, background="black", height=270)
        self.video_area.pack(pady=4, fill="both", expand=True)
        self.video_area.pack_propagate(False)
        self.canvas = tk.Label(self.video_area, background="black")
        self.canvas.place(relx=0.5, rely=0.5, anchor="center")
        if _DND_AVAILABLE:
            self.placeholder = tk.Label(
                self.video_area, background="black", foreground="#888",
                text="Drop a video here\nor use File \u2192 Open",
                justify="center",
            )
            self.placeholder.place(relx=0.5, rely=0.5, anchor="center")
        self.video_area.bind("<Configure>", self._on_area_resize)
        self._resize_job = None
        self._last_box = (0, 0)
        self._register_dnd()

        # position slider
        self.pos = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(
            self, from_=0, to=1, orient="horizontal",
            variable=self.pos, command=self._on_slider,
        )
        self.slider.pack(fill="x")
        self._slider_active = False
        self.slider.bind("<ButtonPress-1>", lambda e: self._set_slider_active(True))
        self.slider.bind("<ButtonRelease-1>", self._on_slider_release)

        self.info_var = tk.StringVar(value="frame 0 / 0    t=0.000s")
        ttk.Label(self, textvariable=self.info_var).pack()

        # transport controls
        row = ttk.Frame(self)
        row.pack(pady=3)
        ttk.Button(row, text="Open", command=self.open).pack(side="left", padx=1)
        ttk.Button(row, text="\u25b6 Play", command=self.play).pack(side="left", padx=1)
        ttk.Button(row, text="\u2759\u2759 Pause", command=self.pause).pack(side="left", padx=1)
        ttk.Button(row, text="\u25a0 Stop", command=self.stop).pack(side="left", padx=1)

        row2 = ttk.Frame(self)
        row2.pack(pady=3)
        ttk.Button(row2, text="\u23ea -1s", command=self.rewind).pack(side="left", padx=1)
        ttk.Button(row2, text="|\u25c0 -1 frame", command=self.step_back).pack(side="left", padx=1)
        ttk.Button(row2, text="+1 frame \u25b6|", command=self.step_forward).pack(side="left", padx=1)
        ttk.Button(row2, text="+1s \u23e9", command=self.fast_forward).pack(side="left", padx=1)

        # marking
        row3 = ttk.Frame(self)
        row3.pack(pady=3)
        ttk.Button(row3, text="Mark 1", command=lambda: self.mark(1)).pack(side="left", padx=1)
        ttk.Button(row3, text="Mark 2", command=lambda: self.mark(2)).pack(side="left", padx=1)
        self.mark_var = tk.StringVar(value="mark1: -    mark2: -")
        ttk.Label(self, textvariable=self.mark_var).pack()

    # --- helpers -----------------------------------------------------------
    def _set_slider_active(self, val):
        self._slider_active = val

    def loaded(self):
        return self.player.path is not None

    def open(self):
        path = filedialog.askopenfilename(
            title="Open video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        self.load_path(path)

    def load_path(self, path):
        """Open a specific file path (used by the menu and Open button)."""
        if not path:
            return
        try:
            self.player.open(path)
        except Exception as exc:  # noqa: BLE001
            self.info_var.set(f"error: {exc}")
            return
        self.name_var.set(path.rsplit("/", 1)[-1])
        if getattr(self, "placeholder", None) is not None:
            self.placeholder.place_forget()
        self._populate_audio_tracks()
        self.slider.configure(to=max(self.player.total_frames - 1, 1))
        self.mark1 = self.mark2 = None
        self._refresh_marks()
        self.app.update_coefficients()
        # Size decode to the current display area before spawning frames.
        box = self._current_box()
        if box[0] > 1 and box[1] > 1:
            self._last_box = box
            self.player.set_display_box(*box)
            self.player.rescale()
        self._show_one()  # display first frame
        self._update_info()

    def play(self):
        if self.loaded():
            self.player.playing = True

    def pause(self):
        self.player.playing = False

    def stop(self):
        if not self.loaded():
            return
        self.player.playing = False
        self.player.seek(0)
        self._show_one()
        self._update_info()

    def step_forward(self):
        if not self.loaded():
            return
        self.player.playing = False
        self._show_one(timeout=1.0)
        self._update_info()

    def step_back(self):
        if not self.loaded():
            return
        self.player.playing = False
        target = self.player.current_index - 1
        self.player.seek(target)
        self._show_one(timeout=1.0)
        self._update_info()

    def _jump_frames(self, delta):
        if not self.loaded():
            return
        self.player.playing = False
        self.player.seek(self.player.current_index + delta)
        self._show_one(timeout=1.0)
        self._update_info()

    def fast_forward(self):
        # jump forward by ~1 second worth of frames
        self._jump_frames(max(1, round(self.player.fps)))

    def rewind(self):
        # jump back by ~1 second worth of frames
        self._jump_frames(-max(1, round(self.player.fps)))

    def _on_slider(self, _value):
        # live label update while dragging; actual seek on release
        if self._slider_active and self.loaded():
            frame = int(float(self.pos.get()))
            self.info_var.set(
                f"frame {frame} / {self.player.total_frames - 1}    "
                f"t={self.player.time_at(frame):.3f}s  (release to seek)"
            )

    def _on_slider_release(self, _event):
        self._slider_active = False
        if not self.loaded():
            return
        self.player.playing = False
        frame = int(float(self.pos.get()))
        self.player.seek(frame)
        self._show_one(timeout=1.0)
        self._update_info()

    def mark(self, which):
        if not self.loaded():
            return
        if which == 1:
            self.mark1 = self.player.current_index
        else:
            self.mark2 = self.player.current_index
        self._refresh_marks()
        self.app.update_coefficients()

    def _populate_audio_tracks(self):
        tracks = self.player.audio_tracks
        if not tracks:
            self.audio_combo.configure(values=[], state="disabled")
            self.audio_var.set("(no audio tracks)")
            return
        labels = [t["label"] for t in tracks]
        self.audio_combo.configure(values=labels, state="readonly")
        self.audio_var.set(labels[0])
        self.player.audio_index = tracks[0]["index"]

    def _on_audio_selected(self, _event=None):
        idx = self.audio_combo.current()
        if 0 <= idx < len(self.player.audio_tracks):
            self.player.audio_index = self.player.audio_tracks[idx]["index"]

    def _refresh_marks(self):
        def fmt(m):
            if m is None:
                return "-"
            return f"{m} ({self.player.time_at(m):.3f}s)"
        self.mark_var.set(f"mark1: {fmt(self.mark1)}    mark2: {fmt(self.mark2)}")

    # --- rendering ---------------------------------------------------------
    def _show_one(self, timeout=1.0):
        frame = self.player.get_next(block_timeout=timeout)
        if frame is None:
            return False
        _idx, ppm = frame
        self._render(ppm)
        return True

    def _render(self, ppm_bytes):
        img = tk.PhotoImage(data=ppm_bytes, format="ppm")
        self.canvas.configure(image=img)
        self._imgref = img

    # --- drag and drop -----------------------------------------------------
    def _register_dnd(self):
        if not _DND_AVAILABLE:
            return
        for widget in (self.video_area, self.canvas):
            try:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop)
                widget.dnd_bind("<<DropEnter>>", self._on_drop_enter)
                widget.dnd_bind("<<DropLeave>>", self._on_drop_leave)
            except Exception:  # noqa: BLE001
                pass

    def _on_drop_enter(self, _event):
        self.video_area.configure(background="#2d5a88")
        return "copy" if DND_FILES is None else DND_FILES

    def _on_drop_leave(self, _event):
        self.video_area.configure(background="black")

    def _on_drop(self, event):
        self.video_area.configure(background="black")
        path = self._first_dropped_path(event.data)
        if path:
            self.load_path(path)

    @staticmethod
    def _first_dropped_path(data):
        """Parse tkdnd's drop payload into a single file path.

        tkdnd returns a Tcl list; paths with spaces are wrapped in braces,
        e.g. '{/home/a b/clip.mp4} /home/c/other.mp4'. If several files are
        dropped, we take the first.
        """
        if not data:
            return None
        paths = []
        buf = ""
        in_brace = False
        for ch in data:
            if in_brace:
                if ch == "}":
                    in_brace = False
                    paths.append(buf)
                    buf = ""
                else:
                    buf += ch
            elif ch == "{":
                in_brace = True
                buf = ""
            elif ch.isspace():
                if buf:
                    paths.append(buf)
                    buf = ""
            else:
                buf += ch
        if buf:
            paths.append(buf)
        paths = [p for p in paths if p]
        return paths[0] if paths else None

    # --- responsive scaling ------------------------------------------------
    def _current_box(self):
        """Available (w, h) inside the video area, minus a small margin."""
        self.video_area.update_idletasks()
        w = self.video_area.winfo_width() - 4
        h = self.video_area.winfo_height() - 4
        return max(w, 2), max(h, 2)

    def _on_area_resize(self, event):
        # Debounce: only act after resizing settles, and only on real changes.
        box = (max(event.width - 4, 2), max(event.height - 4, 2))
        if box == self._last_box:
            return
        self._last_box = box
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(150, self._apply_resize)

    def _apply_resize(self):
        self._resize_job = None
        if not self.loaded():
            return
        box = self._current_box()
        if self.player.set_display_box(*box):
            was_playing = self.player.playing
            self.player.playing = False
            self.player.rescale()
            self._show_one(timeout=1.0)
            self._update_info()
            self.player.playing = was_playing

    def _update_info(self):
        p = self.player
        self.info_var.set(
            f"frame {p.current_index} / {p.total_frames - 1}    "
            f"t={p.time_at(p.current_index):.3f}s"
        )
        if not self._slider_active:
            self.pos.set(p.current_index)

    def tick(self):
        """Called at ~fps by the app; advances playback if playing."""
        if not self.loaded() or not self.player.playing:
            return
        frame = self.player.get_next(block_timeout=0.0)
        if frame is None:
            return  # decoder hasn't produced the next frame yet; try next tick
        _idx, ppm = frame
        self._render(ppm)
        if self.player.current_index >= self.player.total_frames - 1:
            self.player.playing = False
        self._update_info()


# ----------------------------------------------------------------------------
# Modal ffmpeg log dialog
# ----------------------------------------------------------------------------
class LogDialog(tk.Toplevel):
    """A modal, responsive dialog that streams ffmpeg logs.

    * Grabs input so the main window can't be touched while running.
    * Stays open after completion; the user closes it manually.
    * Log lines arrive from a worker thread via a thread-safe queue and are
      flushed to the Text widget on the UI thread with `after`.
    * Offers Cancel while running and Close when idle/finished.
    """

    def __init__(self, master, title="Exporting audio"):
        super().__init__(master)
        self.title(title)
        self.geometry("760x460")
        self.transient(master)

        self._log_queue = queue.Queue()
        self._job = None
        self._finished = False

        header = ttk.Frame(self, padding=(10, 8, 10, 4))
        header.pack(fill="x")
        self.status_var = tk.StringVar(value="Working...")
        ttk.Label(header, textvariable=self.status_var,
                  font=("TkDefaultFont", 10, "bold")).pack(side="left")

        body = ttk.Frame(self, padding=(10, 0, 10, 6))
        body.pack(fill="both", expand=True)
        self.text = tk.Text(body, wrap="none", height=20,
                            background="#1e1e1e", foreground="#d4d4d4",
                            insertbackground="#d4d4d4", font=("TkFixedFont", 9))
        yscroll = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        xscroll = ttk.Scrollbar(body, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        footer = ttk.Frame(self, padding=(10, 0, 10, 10))
        footer.pack(fill="x")
        self.cancel_btn = ttk.Button(footer, text="Cancel", command=self._on_cancel)
        self.cancel_btn.pack(side="right")
        self.close_btn = ttk.Button(footer, text="Close", command=self._on_close,
                                    state="disabled")
        self.close_btn.pack(side="right", padx=(0, 6))

        # modal behavior
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)
        self.grab_set()
        self._poll_logs()

    def attach_job(self, job):
        self._job = job

    # --- logging (thread-safe) --------------------------------------------
    def enqueue_log(self, line):
        """Called from the worker thread."""
        self._log_queue.put(line)

    def _poll_logs(self):
        try:
            while True:
                line = self._log_queue.get_nowait()
                self.text.insert("end", line)
                self.text.see("end")
        except queue.Empty:
            pass
        if not self._finished:
            self.after(50, self._poll_logs)

    # --- completion --------------------------------------------------------
    def mark_finished(self, ok, message):
        """Called from the worker thread when the job ends."""
        # marshal onto the UI thread
        self.after(0, lambda: self._finish_ui(ok, message))

    def _finish_ui(self, ok, message):
        # drain any remaining log lines first
        try:
            while True:
                self.text.insert("end", self._log_queue.get_nowait())
        except queue.Empty:
            pass
        self.text.see("end")
        self._finished = True
        self.status_var.set(("Done \u2713  " if ok else "Failed \u2717  ") + message)
        self.cancel_btn.configure(state="disabled")
        self.close_btn.configure(state="normal")

    # --- buttons -----------------------------------------------------------
    def _on_cancel(self):
        if self._job is not None and not self._finished:
            self.status_var.set("Cancelling...")
            self._job.cancel()

    def _on_close(self):
        self.grab_release()
        self.destroy()

    def _on_window_close(self):
        # Only allow closing via the window manager once finished; otherwise
        # treat it as a cancel request.
        if self._finished:
            self._on_close()
        else:
            self._on_cancel()


# ----------------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------------
class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        self.master = master
        self.pack(fill="both", expand=True)

        panels = ttk.Frame(self)
        panels.pack(fill="both", expand=True)
        self.left = VideoPanel(panels, self, "Video A  (x / left timeline)")
        self.left.pack(side="left", fill="both", expand=True, padx=4)
        self.right = VideoPanel(panels, self, "Video B  (y / right timeline)")
        self.right.pack(side="left", fill="both", expand=True, padx=4)

        self._build_menu(master)

        # coefficient panel
        coef = ttk.LabelFrame(self, text="Timeline mapping  y = a*x + b  (seconds)", padding=8)
        coef.pack(fill="x", pady=6)
        self.coef_var = tk.StringVar(value="Mark two frames in each video to compute a and b.")
        ttk.Label(coef, textvariable=self.coef_var, font=("TkFixedFont", 10)).pack(anchor="w")
        self.hint_var = tk.StringVar(value="")
        ttk.Label(coef, textvariable=self.hint_var, foreground="#555").pack(anchor="w")

        self._build_export_controls(coef)

        # keyboard shortcuts
        master.bind("<Left>", lambda e: self.left.step_back())
        master.bind("<Right>", lambda e: self.left.step_forward())
        master.bind("<Shift-Left>", lambda e: self.left.rewind())
        master.bind("<Shift-Right>", lambda e: self.left.fast_forward())
        master.bind("<comma>", lambda e: self.right.step_back())
        master.bind("<period>", lambda e: self.right.step_forward())
        master.bind("<less>", lambda e: self.right.rewind())
        master.bind("<greater>", lambda e: self.right.fast_forward())
        master.bind("<space>", self._toggle_both)

        self.coef_a = None
        self.coef_b = None
        self._update_export_buttons()

        self._alive = True
        self._schedule_tick()

    def _build_export_controls(self, parent):
        opts = ttk.Frame(parent)
        opts.pack(fill="x", pady=(8, 2))

        # audio codec options
        ttk.Label(opts, text="Codec:").pack(side="left")
        self.codec_var = tk.StringVar(value="aac")
        codec_combo = ttk.Combobox(
            opts, textvariable=self.codec_var, width=8, state="readonly",
            values=["aac", "libmp3lame", "libopus", "ac3", "flac", "pcm_s16le"],
        )
        codec_combo.pack(side="left", padx=(4, 12))

        ttk.Label(opts, text="Bitrate:").pack(side="left")
        self.bitrate_var = tk.StringVar(value="320k")
        bitrate_combo = ttk.Combobox(
            opts, textvariable=self.bitrate_var, width=7,
            values=["128k", "192k", "256k", "320k", "384k"],
        )
        bitrate_combo.pack(side="left", padx=(4, 12))

        ttk.Label(opts, text="Channels:").pack(side="left")
        self.channels_var = tk.StringVar(value="2")
        channels_combo = ttk.Combobox(
            opts, textvariable=self.channels_var, width=4, state="readonly",
            values=["1", "2", "6"],
        )
        channels_combo.pack(side="left", padx=(4, 12))

        self.normalize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Normalize (loudnorm)",
                        variable=self.normalize_var).pack(side="left", padx=(0, 12))

        self.preserve_pitch_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Preserve pitch",
                        variable=self.preserve_pitch_var).pack(side="left")

        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(6, 0))
        self.export_a_btn = ttk.Button(
            btns, text="Save A's audio synced to B \u2192",
            command=lambda: self.export_audio("A_to_B"), state="disabled")
        self.export_a_btn.pack(side="left", padx=(0, 6))
        self.export_b_btn = ttk.Button(
            btns, text="\u2190 Save B's audio synced to A",
            command=lambda: self.export_audio("B_to_A"), state="disabled")
        self.export_b_btn.pack(side="left")

    def _build_menu(self, master):
        menubar = tk.Menu(master)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="Open Video A...", accelerator="Ctrl+1",
            command=self.left.open,
        )
        file_menu.add_command(
            label="Open Video B...", accelerator="Ctrl+2",
            command=self.right.open,
        )
        file_menu.add_command(
            label="Open Both Videos...", accelerator="Ctrl+O",
            command=self.open_both,
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._exit)
        menubar.add_cascade(label="File", menu=file_menu)
        master.config(menu=menubar)

        master.bind("<Control-Key-1>", lambda e: self.left.open())
        master.bind("<Control-Key-2>", lambda e: self.right.open())
        master.bind("<Control-o>", lambda e: self.open_both())

    def open_both(self):
        """Pick two files in one dialog (or fall back to two prompts)."""
        paths = filedialog.askopenfilenames(
            title="Select two videos (A then B)",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"),
                       ("All files", "*.*")],
        )
        paths = list(paths)
        if len(paths) >= 2:
            self.left.load_path(paths[0])
            self.right.load_path(paths[1])
        elif len(paths) == 1:
            # Only one selected: load into A, then prompt for B.
            self.left.load_path(paths[0])
            self.right.open()
        else:
            # Nothing selected via multi-select; fall back to sequential prompts.
            self.left.open()
            if self.left.loaded():
                self.right.open()

    def _exit(self):
        self.shutdown()
        self.master.destroy()

    def shutdown(self):
        """Stop tick loop and kill both decoder subprocesses."""
        self._alive = False
        for panel in (self.left, self.right):
            panel.player.close()

    def _toggle_both(self, _event=None):
        want = not (self.left.player.playing or self.right.player.playing)
        for panel in (self.left, self.right):
            if panel.loaded():
                panel.player.playing = want

    def _schedule_tick(self):
        if not self._alive:
            return
        # tick rate based on the fastest loaded video, capped to a sane range
        fps = 30.0
        for panel in (self.left, self.right):
            if panel.loaded() and panel.player.fps > 0:
                fps = max(fps, panel.player.fps)
        delay = int(1000 / min(max(fps, 5.0), 60.0))
        self.left.tick()
        self.right.tick()
        self.after(delay, self._schedule_tick)

    def update_coefficients(self):
        l, r = self.left.player, self.right.player
        self.coef_a = None
        self.coef_b = None
        self._update_export_buttons()
        if None in (self.left.mark1, self.left.mark2,
                    self.right.mark1, self.right.mark2):
            self.coef_var.set("Mark two frames in each video to compute a and b.")
            self.hint_var.set("")
            return
        x1 = l.time_at(self.left.mark1)
        x2 = l.time_at(self.left.mark2)
        y1 = r.time_at(self.right.mark1)
        y2 = r.time_at(self.right.mark2)
        if x2 == x1:
            self.coef_var.set("Cannot compute: the two marks in Video A are the same frame.")
            self.hint_var.set("")
            return
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        self.coef_a = a
        self.coef_b = b
        self.coef_var.set(
            f"a = {a:.6f}    b = {b:.6f} s\n"
            f"points: ({x1:.3f}, {y1:.3f})  ->  ({x2:.3f}, {y2:.3f})"
        )
        self.hint_var.set(
            "a = speed ratio of B relative to A;  b = offset (s) of B at A's t=0. "
            "Audio: A\u2192B uses tempo a, offset b;  B\u2192A uses tempo 1/a, offset -b/a."
        )
        self._update_export_buttons()

    def _update_export_buttons(self):
        ready = (getattr(self, "coef_a", None) is not None
                 and self.coef_a != 0)
        a_ok = ready and self.left.loaded() and self.left.player.audio_tracks
        b_ok = ready and self.right.loaded() and self.right.player.audio_tracks
        self.export_a_btn.configure(state="normal" if a_ok else "disabled")
        self.export_b_btn.configure(state="normal" if b_ok else "disabled")

    def export_audio(self, direction):
        a, b = getattr(self, "coef_a", None), getattr(self, "coef_b", None)
        if a is None or a == 0:
            messagebox.showwarning("Not ready",
                                   "Compute coefficients first (mark 2 frames each).")
            return

        if direction == "A_to_B":
            src_panel, target_panel = self.left, self.right
            stretch, offset = a, b
            default_name = "A_synced_to_B"
        else:  # B_to_A
            src_panel, target_panel = self.right, self.left
            stretch, offset = 1.0 / a, -b / a
            default_name = "B_synced_to_A"

        if not src_panel.loaded() or not src_panel.player.audio_tracks:
            messagebox.showwarning("No audio",
                                   "The source video has no audio track.")
            return

        target_duration = probe_duration(target_panel.player.path)
        if target_duration <= 0:
            messagebox.showwarning(
                "Unknown duration",
                "Could not determine the target video's duration.")
            return

        codec = self.codec_var.get()
        ext = {"aac": ".m4a", "libmp3lame": ".mp3", "libopus": ".opus",
               "ac3": ".ac3", "flac": ".flac", "pcm_s16le": ".wav"}.get(codec, ".m4a")
        out_path = filedialog.asksaveasfilename(
            title="Save synced audio as",
            initialfile=default_name + ext,
            defaultextension=ext,
        )
        if not out_path:
            return

        dialog = LogDialog(self.master, title=f"Exporting audio ({direction})")

        def done_cb(ok, msg):
            dialog.mark_finished(ok, msg)

        job = AudioExportJob(
            src_path=src_panel.player.path,
            src_audio_index=src_panel.player.audio_index or 0,
            stretch=stretch,
            offset=offset,
            target_duration=target_duration,
            out_path=out_path,
            preserve_pitch=self.preserve_pitch_var.get(),
            normalize=self.normalize_var.get(),
            codec=codec,
            bitrate=self.bitrate_var.get(),
            channels=int(self.channels_var.get()),
            log_cb=dialog.enqueue_log,
            done_cb=done_cb,
        )
        dialog.attach_job(job)
        job.start()


def main():
    root = TkinterDnD.Tk() if _DND_AVAILABLE else tk.Tk()
    root.title("Dual Video Frame Sync")
    root.geometry("1080x760")
    app = App(root)

    def on_close():
        # Tear down both decoder pipes so no ffmpeg children are orphaned.
        app.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
