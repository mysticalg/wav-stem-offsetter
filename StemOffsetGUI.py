# StemOffsetGUI.py
# Windows GUI: Drag & drop WAV -> Demucs stem extraction (detect dynamically)
# -> optional Noise Reduction + De-ess + EQ per stem -> per-stem offsets -> render + Save WAV
#
# Requires:
#   pip install PySide6 demucs soundfile numpy scipy
#
# Notes:
# - Demucs install on Windows can be finicky (torch/torchaudio/torchcodec/ffmpeg combos).
# - This app assumes Demucs can run successfully in your venv.

import os
import sys
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QComboBox, QCheckBox, QScrollArea
)

# ---- SciPy DSP (required for EQ + STFT tools) ----
try:
    from scipy.signal import butter, sosfiltfilt, stft, istft
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# --------------------------
# Audio helpers
# --------------------------
def apply_offset(signal: np.ndarray, offset_samples: int) -> np.ndarray:
    """
    Positive offset_samples delays the audio (pads at start).
    Negative offset_samples advances the audio (cuts from start).
    Keeps same length by padding/cutting accordingly.
    signal shape: (n_samples, n_channels)
    """
    if offset_samples == 0:
        return signal

    n = signal.shape[0]

    if offset_samples > 0:
        pad = np.zeros((offset_samples, signal.shape[1]), dtype=signal.dtype)
        shifted = np.concatenate([pad, signal], axis=0)[:n]
    else:
        k = abs(offset_samples)
        shifted = signal[k:]
        if shifted.shape[0] < n:
            pad = np.zeros((n - shifted.shape[0], signal.shape[1]), dtype=signal.dtype)
            shifted = np.concatenate([shifted, pad], axis=0)

    return shifted


def run_demucs(input_wav: Path, out_dir: Path, model: str, log_cb=None):
    """
    Runs demucs and returns stems folder:
      out_dir/<model>/<track_name>/*.wav
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(out_dir), str(input_wav)]
    if log_cb:
        log_cb("Running Demucs:\n  " + " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    if proc.stdout:
        for line in proc.stdout:
            if log_cb:
                log_cb(line.rstrip())

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Demucs failed with exit code {ret}")

    track_name = input_wav.stem
    stems_dir = out_dir / model / track_name
    if stems_dir.exists():
        return stems_dir

    # Fallback: find a directory that matches track_name
    candidates = [p for p in out_dir.glob(f"**/{track_name}") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Could not find stems output folder under: {out_dir}")
    return candidates[0]


def load_stems(stems_dir: Path):
    """
    Loads stem WAVs into dict {stem_name: audio ndarray}, returns (stems_audio, sr)
    Reads ALL .wav files in folder, stem name = file stem.
    """
    stem_files = {p.stem: p for p in stems_dir.glob("*.wav")}
    if not stem_files:
        raise FileNotFoundError(f"No stem wavs found in {stems_dir}")

    stems_audio = {}
    sr = None
    max_len = 0

    for name, path in stem_files.items():
        audio, sr_i = sf.read(path, always_2d=True)  # (n, ch)
        # Convert to float32 for consistent DSP / performance
        audio = audio.astype(np.float32, copy=False)

        if sr is None:
            sr = sr_i
        elif sr_i != sr:
            raise ValueError(f"Sample rate mismatch in stem {path.name}: {sr_i} vs {sr}")

        stems_audio[name] = audio
        max_len = max(max_len, audio.shape[0])

    # Pad all to same length
    for name, audio in stems_audio.items():
        if audio.shape[0] < max_len:
            pad = np.zeros((max_len - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
            stems_audio[name] = np.concatenate([audio, pad], axis=0)

    return stems_audio, sr


# --------------------------
# DSP blocks
# --------------------------
def find_quietest_window(audio: np.ndarray, sr: int, window_sec: float = 1.0, hop_sec: float = 0.25):
    """
    Finds the lowest-RMS window (mono-averaged) and returns (start_idx, end_idx) sample indices.
    """
    mono = np.mean(audio, axis=1)
    win = int(max(1, round(window_sec * sr)))
    hop = int(max(1, round(hop_sec * sr)))

    if win >= len(mono):
        return 0, len(mono)

    best_i = 0
    best_rms = float("inf")

    # Quick RMS scan
    for i in range(0, len(mono) - win, hop):
        seg = mono[i:i + win]
        rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
        if rms < best_rms:
            best_rms = rms
            best_i = i

    return best_i, best_i + win


def noise_reduce_spectral(audio: np.ndarray, sr: int, noise_window_sec: float = 1.0,
                          strength: float = 1.0, floor_db: float = -60.0,
                          nperseg: int = 2048, noverlap: int = 1536):
    """
    Simple spectral subtraction using a noise profile from the quietest segment.
    Returns: (clean_audio, (noise_start_sec, noise_end_sec))
    """
    if not SCIPY_OK:
        raise RuntimeError("SciPy is required for noise reduction. Install with: pip install scipy")

    start, end = find_quietest_window(audio, sr, window_sec=noise_window_sec, hop_sec=0.25)
    noise = audio[start:end, :]

    # STFT expects (channels, samples) for our usage below
    f, t, Z = stft(audio.T, fs=sr, nperseg=nperseg, noverlap=noverlap)
    _, _, Zn = stft(noise.T, fs=sr, nperseg=nperseg, noverlap=noverlap)

    # (ch, f, t) -> (f, t, ch)
    Z = np.transpose(Z, (1, 2, 0))
    Zn = np.transpose(Zn, (1, 2, 0))

    mag = np.abs(Z)
    phase = np.exp(1j * np.angle(Z))

    # noise profile: average magnitude across time frames of noise segment
    noise_mag = np.mean(np.abs(Zn), axis=1, keepdims=True)  # (f, 1, ch)

    cleaned_mag = np.maximum(0.0, mag - float(strength) * noise_mag)

    # Prevent “watery” artifacts by imposing a floor
    floor = 10 ** (float(floor_db) / 20.0)
    cleaned_mag = np.maximum(cleaned_mag, floor)

    Zc = cleaned_mag * phase

    Zc_back = np.transpose(Zc, (2, 0, 1))  # (ch, f, t)
    _, y = istft(Zc_back, fs=sr, nperseg=nperseg, noverlap=noverlap)
    y = y.T.astype(np.float32, copy=False)

    # Length match (istft can differ by a few samples)
    if y.shape[0] < audio.shape[0]:
        pad = np.zeros((audio.shape[0] - y.shape[0], y.shape[1]), dtype=y.dtype)
        y = np.concatenate([y, pad], axis=0)
    elif y.shape[0] > audio.shape[0]:
        y = y[:audio.shape[0], :]

    return y, (start / sr, end / sr)


def de_ess_stft(audio: np.ndarray, sr: int,
                f_lo: float = 8000.0, f_hi: float = 12000.0,
                thresh_db: float = -25.0, ratio: float = 4.0,
                nperseg: int = 2048, noverlap: int = 1536):
    """
    Simple de-esser / dynamic EQ:
    - Compute band level (8–12k default) per frame
    - Apply compression gain only to that band’s bins.
    """
    if not SCIPY_OK:
        raise RuntimeError("SciPy is required for de-essing. Install with: pip install scipy")

    f, t, Z = stft(audio.T, fs=sr, nperseg=nperseg, noverlap=noverlap)  # (ch, f, t)
    Z = np.transpose(Z, (1, 2, 0))  # (f, t, ch)

    band = (f >= float(f_lo)) & (f <= float(f_hi))
    mag = np.abs(Z)
    eps = 1e-12

    band_level = np.mean(mag[band, :, :], axis=0)  # (t, ch)
    band_db = 20.0 * np.log10(band_level + eps)

    over_db = np.maximum(0.0, band_db - float(thresh_db))
    ratio = max(1.0, float(ratio))
    gr_db = over_db * (1.0 - 1.0 / ratio)
    gain = 10.0 ** (-gr_db / 20.0)

    Z[band, :, :] *= gain[None, :, :]

    Z_back = np.transpose(Z, (2, 0, 1))  # (ch, f, t)
    _, y = istft(Z_back, fs=sr, nperseg=nperseg, noverlap=noverlap)
    y = y.T.astype(np.float32, copy=False)

    # Length match
    if y.shape[0] < audio.shape[0]:
        pad = np.zeros((audio.shape[0] - y.shape[0], y.shape[1]), dtype=y.dtype)
        y = np.concatenate([y, pad], axis=0)
    elif y.shape[0] > audio.shape[0]:
        y = y[:audio.shape[0], :]

    return y


def lowpass_sos(audio: np.ndarray, sr: int, cutoff_hz: float = 16000.0, order: int = 3):
    """
    Butterworth low-pass (filtfilt via SOS).
    """
    if not SCIPY_OK:
        raise RuntimeError("SciPy is required for EQ. Install with: pip install scipy")

    cutoff_hz = float(cutoff_hz)
    order = int(order)
    cutoff_hz = min(max(1000.0, cutoff_hz), (sr / 2.0) - 100.0)
    order = min(max(1, order), 8)

    sos = butter(order, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, audio, axis=0).astype(np.float32, copy=False)
    return y


def process_stem(audio: np.ndarray, sr: int, opts: dict, log_cb=None):
    """
    Applies (optional) noise reduction, de-ess, EQ to a stem.
    opts keys:
      enable_nr, nr_window, nr_strength, nr_floor_db
      enable_deess, deess_flo, deess_fhi, deess_thresh_db, deess_ratio
      enable_eq, eq_cutoff_hz, eq_order
    """
    y = audio

    if opts.get("enable_nr", False):
        y, (ns, ne) = noise_reduce_spectral(
            y, sr,
            noise_window_sec=float(opts.get("nr_window", 1.0)),
            strength=float(opts.get("nr_strength", 1.0)),
            floor_db=float(opts.get("nr_floor_db", -60.0)),
        )
        if log_cb:
            log_cb(f"Noise reduction: profile window {ns:.2f}s–{ne:.2f}s")

    if opts.get("enable_deess", False):
        y = de_ess_stft(
            y, sr,
            f_lo=float(opts.get("deess_flo", 8000.0)),
            f_hi=float(opts.get("deess_fhi", 12000.0)),
            thresh_db=float(opts.get("deess_thresh_db", -25.0)),
            ratio=float(opts.get("deess_ratio", 4.0)),
        )
        if log_cb:
            log_cb("De-ess applied")

    if opts.get("enable_eq", False):
        y = lowpass_sos(
            y, sr,
            cutoff_hz=float(opts.get("eq_cutoff_hz", 16000.0)),
            order=int(opts.get("eq_order", 3)),
        )
        if log_cb:
            log_cb("Low-pass EQ applied")

    return y


# --------------------------
# Worker threads
# --------------------------
class DetectWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    detected = Signal(object, int, list, str)  # stems_audio, sr, names, temp_root
    failed = Signal(str)

    def __init__(self, input_wav: Path, model: str, keep_temp: bool):
        super().__init__()
        self.input_wav = input_wav
        self.model = model
        self.keep_temp = keep_temp

    def run(self):
        temp_root = None
        try:
            self.progress.emit(5)
            temp_root = Path(tempfile.mkdtemp(prefix="stem_offset_gui_"))
            demucs_out = temp_root / "demucs_out"

            self.progress.emit(10)
            stems_dir = run_demucs(self.input_wav, demucs_out, self.model, log_cb=self.log.emit)

            self.progress.emit(70)
            stems_audio, sr = load_stems(stems_dir)
            names = sorted(stems_audio.keys(), key=lambda s: s.lower())

            self.progress.emit(100)
            self.detected.emit(stems_audio, sr, names, str(temp_root))

        except Exception as e:
            self.failed.emit(str(e))
            if temp_root and temp_root.exists() and not self.keep_temp:
                shutil.rmtree(temp_root, ignore_errors=True)


class RenderWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, stems_audio: dict, sr: int, offsets_ms: dict,
                 master_gain: float, out_wav: Path, dsp_opts: dict):
        super().__init__()
        self.stems_audio = stems_audio
        self.sr = sr
        self.offsets_ms = offsets_ms
        self.master_gain = float(master_gain)
        self.out_wav = out_wav
        self.dsp_opts = dsp_opts

    def run(self):
        try:
            if not SCIPY_OK and (
                self.dsp_opts.get("enable_nr") or self.dsp_opts.get("enable_deess") or self.dsp_opts.get("enable_eq")
            ):
                raise RuntimeError("SciPy is required for EQ/De-ess/Noise Reduction. Install with: pip install scipy")

            stem_names = list(self.stems_audio.keys())
            n = len(stem_names)
            if n == 0:
                raise RuntimeError("No stems loaded.")

            self.progress.emit(5)
            self.log.emit(f"Rendering {n} stems…")

            # Convert offsets to samples
            offsets_samples = {
                name: int(round((float(self.offsets_ms.get(name, 0.0)) / 1000.0) * self.sr))
                for name in stem_names
            }

            # Process each stem, then offset, then sum
            mix = None
            for idx, name in enumerate(stem_names, start=1):
                self.log.emit(f"[{idx}/{n}] Stem: {name}")

                y = self.stems_audio[name]

                # DSP (optional)
                def stem_log(msg):
                    self.log.emit(f"    {msg}")

                y = process_stem(y, self.sr, self.dsp_opts, log_cb=stem_log)

                # Offset
                y = apply_offset(y, offsets_samples.get(name, 0))

                if mix is None:
                    mix = np.zeros_like(y, dtype=np.float32)
                mix += y

                # progress 5..85
                self.progress.emit(int(5 + (80 * idx / n)))

            # Prevent clipping, then gain
            self.log.emit("Normalizing / applying master gain…")
            peak = float(np.max(np.abs(mix))) if mix is not None and mix.size else 0.0
            if peak > 1.0:
                mix = mix / peak

            mix *= self.master_gain

            self.progress.emit(92)
            self.out_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(self.out_wav, mix, self.sr)
            self.progress.emit(100)

            self.done.emit(f"Saved: {self.out_wav}")

        except Exception as e:
            self.failed.emit(str(e))


# --------------------------
# Main Window
# --------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stem Offset Remixer + DSP (Demucs)")

        self.setAcceptDrops(True)

        self.current_wav: Path | None = None

        self.detected_stems_audio = None  # dict: {stem_name: ndarray}
        self.detected_sr = None
        self.detected_stem_names = []
        self.demucs_temp_root = None

        self.detect_worker: DetectWorker | None = None
        self.render_worker: RenderWorker | None = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.drop_label = QLabel("Drag & drop a WAV file here\n(or click “Open WAV…”)")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet(
            "QLabel { border: 2px dashed #777; padding: 24px; font-size: 14px; }"
        )
        layout.addWidget(self.drop_label)

        # Buttons
        btn_row = QHBoxLayout()
        self.open_btn = QPushButton("Open WAV…")
        self.detect_btn = QPushButton("Extract Stems / Detect…")
        self.render_btn = QPushButton("Render + Save As…")

        self.detect_btn.setEnabled(False)
        self.render_btn.setEnabled(False)

        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.detect_btn)
        btn_row.addWidget(self.render_btn)
        layout.addLayout(btn_row)

        # Options group
        opt_box = QGroupBox("Options")
        opt_layout = QFormLayout(opt_box)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["htdemucs", "htdemucs_ft", "mdx_extra", "mdx"])
        self.model_combo.setCurrentText("htdemucs")
        opt_layout.addRow("Demucs model", self.model_combo)

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.10, 1.00)
        self.gain_spin.setSingleStep(0.05)
        self.gain_spin.setValue(0.95)
        opt_layout.addRow("Master gain", self.gain_spin)

        self.keep_stems_cb = QCheckBox("Keep temporary Demucs files (debug)")
        self.keep_stems_cb.setChecked(False)
        opt_layout.addRow("", self.keep_stems_cb)

        layout.addWidget(opt_box)

        # DSP group
        dsp_box = QGroupBox("DSP (applied per stem before offsets)")
        dsp_layout = QFormLayout(dsp_box)

        self.enable_eq = QCheckBox("Enable EQ (Low-pass)")
        self.enable_eq.setChecked(True)
        dsp_layout.addRow(self.enable_eq)

        self.eq_cutoff = QSpinBox()
        self.eq_cutoff.setRange(12000, 20000)
        self.eq_cutoff.setValue(16000)
        dsp_layout.addRow("Low-pass cutoff (Hz)", self.eq_cutoff)

        self.eq_order = QSpinBox()
        self.eq_order.setRange(1, 8)
        self.eq_order.setValue(3)
        dsp_layout.addRow("Low-pass order (1–8)", self.eq_order)

        self.enable_deess = QCheckBox("Enable De-ess / Dynamic EQ (8–12 kHz)")
        self.enable_deess.setChecked(False)
        dsp_layout.addRow(self.enable_deess)

        self.deess_thresh = QDoubleSpinBox()
        self.deess_thresh.setRange(-80.0, 0.0)
        self.deess_thresh.setSingleStep(1.0)
        self.deess_thresh.setValue(-25.0)
        dsp_layout.addRow("De-ess threshold (dB)", self.deess_thresh)

        self.deess_ratio = QDoubleSpinBox()
        self.deess_ratio.setRange(1.0, 20.0)
        self.deess_ratio.setSingleStep(0.5)
        self.deess_ratio.setValue(4.0)
        dsp_layout.addRow("De-ess ratio", self.deess_ratio)

        self.enable_nr = QCheckBox("Enable Noise Reduction (auto noise profile)")
        self.enable_nr.setChecked(False)
        dsp_layout.addRow(self.enable_nr)

        self.nr_window = QDoubleSpinBox()
        self.nr_window.setRange(0.2, 5.0)
        self.nr_window.setSingleStep(0.1)
        self.nr_window.setValue(1.0)
        dsp_layout.addRow("Noise profile window (sec)", self.nr_window)

        self.nr_strength = QDoubleSpinBox()
        self.nr_strength.setRange(0.0, 3.0)
        self.nr_strength.setSingleStep(0.1)
        self.nr_strength.setValue(1.0)
        dsp_layout.addRow("NR strength", self.nr_strength)

        self.nr_floor = QDoubleSpinBox()
        self.nr_floor.setRange(-90.0, -30.0)
        self.nr_floor.setSingleStep(1.0)
        self.nr_floor.setValue(-60.0)
        dsp_layout.addRow("NR floor (dB)", self.nr_floor)

        layout.addWidget(dsp_box)

        # Offsets group (dynamic) with scroll
        self.offsets_box = QGroupBox("Offsets per stem (ms) — detected dynamically")
        offsets_outer = QVBoxLayout(self.offsets_box)

        self.offsets_scroll = QScrollArea()
        self.offsets_scroll.setWidgetResizable(True)
        offsets_outer.addWidget(self.offsets_scroll)

        self.offsets_container = QWidget()
        self.offsets_form = QFormLayout(self.offsets_container)
        self.offsets_scroll.setWidget(self.offsets_container)

        self.offset_spins = {}  # stem_name -> QSpinBox

        rand_row = QHBoxLayout()
        self.rand_min = QSpinBox()
        self.rand_min.setRange(1, 200)
        self.rand_min.setValue(1)
        self.rand_max = QSpinBox()
        self.rand_max.setRange(1, 200)
        self.rand_max.setValue(200)
        self.randomize_btn = QPushButton("Randomize")
        self.unique_rand_cb = QCheckBox("Unique per stem")
        self.unique_rand_cb.setChecked(True)

        rand_row.addWidget(QLabel("Random range"))
        rand_row.addWidget(self.rand_min)
        rand_row.addWidget(QLabel("to"))
        rand_row.addWidget(self.rand_max)
        rand_row.addWidget(QLabel("ms"))
        rand_row.addWidget(self.unique_rand_cb)
        rand_row.addWidget(self.randomize_btn)
        offsets_outer.addLayout(rand_row)

        layout.addWidget(self.offsets_box)

        # Progress + log
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(180)
        layout.addWidget(self.log_box)

        # Wiring
        self.open_btn.clicked.connect(self.open_wav)
        self.detect_btn.clicked.connect(self.extract_and_detect)
        self.render_btn.clicked.connect(self.render_and_save)
        self.randomize_btn.clicked.connect(self.randomize_offsets)

        if not SCIPY_OK:
            self.log("WARNING: SciPy not found. EQ/De-ess/Noise Reduction will not work until you: pip install scipy")

        self._rebuild_offsets_ui([])

    # ---- Drag & drop ----
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(".wav"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if path.suffix.lower() != ".wav":
            return
        self.set_current_wav(path)

    # ---- UI helpers ----
    def log(self, text: str):
        self.log_box.append(text)

    def set_current_wav(self, path: Path):
        self.current_wav = path
        self.drop_label.setText(f"Loaded:\n{path}")
        self.progress.setValue(0)
        self.log_box.clear()
        self.log(f"Loaded WAV: {path}")

        self.detect_btn.setEnabled(True)
        self.render_btn.setEnabled(False)

        self.detected_stems_audio = None
        self.detected_sr = None
        self.detected_stem_names = []
        self._rebuild_offsets_ui([])

    def open_wav(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open WAV", "", "WAV files (*.wav)"
        )
        if file_path:
            self.set_current_wav(Path(file_path))

    def _rebuild_offsets_ui(self, stem_names):
        # Clear old rows
        while self.offsets_form.rowCount():
            self.offsets_form.removeRow(0)

        self.offset_spins.clear()

        if not stem_names:
            self.offsets_form.addRow(QLabel("No stems detected yet. Click “Extract Stems / Detect…”"), QLabel(""))
            return

        for stem in stem_names:
            sp = QSpinBox()
            sp.setRange(0, 200)     # set to 1..200 if you want to forbid 0
            sp.setValue(10)
            self.offset_spins[stem] = sp
            self.offsets_form.addRow(stem, sp)

    def randomize_offsets(self):
        if not self.offset_spins:
            self.log("No stems detected yet — click Extract Stems / Detect first.")
            return

        lo = self.rand_min.value()
        hi = self.rand_max.value()
        if lo > hi:
            lo, hi = hi, lo

        stems = list(self.offset_spins.keys())
        count = len(stems)

        if self.unique_rand_cb.isChecked() and (hi - lo + 1) >= count:
            vals = random.sample(range(lo, hi + 1), count)
        else:
            vals = [random.randint(lo, hi) for _ in stems]

        for stem, val in zip(stems, vals):
            self.offset_spins[stem].setValue(val)

        self.log(f"Randomized offsets ({lo}..{hi} ms).")

    def gather_offsets(self) -> dict:
        return {stem: sp.value() for stem, sp in self.offset_spins.items()}

    def gather_dsp_opts(self) -> dict:
        return {
            "enable_eq": self.enable_eq.isChecked(),
            "eq_cutoff_hz": self.eq_cutoff.value(),
            "eq_order": self.eq_order.value(),

            "enable_deess": self.enable_deess.isChecked(),
            "deess_flo": 8000.0,
            "deess_fhi": 12000.0,
            "deess_thresh_db": self.deess_thresh.value(),
            "deess_ratio": self.deess_ratio.value(),

            "enable_nr": self.enable_nr.isChecked(),
            "nr_window": self.nr_window.value(),
            "nr_strength": self.nr_strength.value(),
            "nr_floor_db": self.nr_floor.value(),
        }

    # ---- Detect stems ----
    def extract_and_detect(self):
        if not self.current_wav or not self.current_wav.exists():
            QMessageBox.warning(self, "No input", "Please load a WAV first.")
            return

        self._lock_ui(detecting=True)
        self.progress.setValue(0)
        self.log_box.clear()
        self.log("Extracting stems with Demucs...")

        self.detect_worker = DetectWorker(
            input_wav=self.current_wav,
            model=self.model_combo.currentText(),
            keep_temp=self.keep_stems_cb.isChecked(),
        )
        self.detect_worker.progress.connect(self.progress.setValue)
        self.detect_worker.log.connect(self.log)
        self.detect_worker.detected.connect(self.on_detected)
        self.detect_worker.failed.connect(self.on_detect_failed)
        self.detect_worker.start()

    def on_detected(self, stems_audio, sr, names, temp_root_str):
        # Clean old temp root if not keeping
        if self.demucs_temp_root and not self.keep_stems_cb.isChecked():
            try:
                shutil.rmtree(self.demucs_temp_root, ignore_errors=True)
            except Exception:
                pass

        self.demucs_temp_root = temp_root_str
        self.detected_stems_audio = stems_audio
        self.detected_sr = sr
        self.detected_stem_names = names

        self.log(f"Detected stems ({len(names)}): {names}")
        self._rebuild_offsets_ui(names)

        self._unlock_ui(detected=True)

    def on_detect_failed(self, err):
        self.log("ERROR: " + err)
        QMessageBox.critical(self, "Detect failed", err)
        self._unlock_ui(detected=False)

    # ---- Render ----
    def render_and_save(self):
        if not self.detected_stems_audio or not self.detected_sr:
            QMessageBox.warning(self, "No stems", "Click Extract Stems / Detect first.")
            return

        default_name = self.current_wav.with_name(self.current_wav.stem + "_shifted.wav")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed WAV As", str(default_name), "WAV files (*.wav)"
        )
        if not out_path:
            return

        offsets = self.gather_offsets()
        dsp_opts = self.gather_dsp_opts()

        self._lock_ui(detecting=False)
        self.progress.setValue(0)
        self.log("Starting render…")

        self.render_worker = RenderWorker(
            stems_audio=self.detected_stems_audio,
            sr=self.detected_sr,
            offsets_ms=offsets,
            master_gain=self.gain_spin.value(),
            out_wav=Path(out_path),
            dsp_opts=dsp_opts,
        )
        self.render_worker.progress.connect(self.progress.setValue)
        self.render_worker.log.connect(self.log)
        self.render_worker.done.connect(self.on_render_done)
        self.render_worker.failed.connect(self.on_render_failed)
        self.render_worker.start()

    def on_render_done(self, msg: str):
        self.log(msg)
        QMessageBox.information(self, "Done", msg)
        self._unlock_ui(detected=True)

    def on_render_failed(self, err: str):
        self.log("ERROR: " + err)
        QMessageBox.critical(self, "Render failed", err)
        self._unlock_ui(detected=True)

    # ---- UI lock/unlock ----
    def _lock_ui(self, detecting: bool):
        self.open_btn.setEnabled(False)
        self.detect_btn.setEnabled(False)
        self.render_btn.setEnabled(False)
        self.randomize_btn.setEnabled(False)

        self.model_combo.setEnabled(False)
        self.gain_spin.setEnabled(False)
        self.keep_stems_cb.setEnabled(False)

        # DSP controls
        for w in (self.enable_eq, self.eq_cutoff, self.eq_order,
                  self.enable_deess, self.deess_thresh, self.deess_ratio,
                  self.enable_nr, self.nr_window, self.nr_strength, self.nr_floor):
            w.setEnabled(False)

        # Offset spins
        for sp in self.offset_spins.values():
            sp.setEnabled(False)

    def _unlock_ui(self, detected: bool):
        self.open_btn.setEnabled(True)
        self.detect_btn.setEnabled(self.current_wav is not None)
        self.render_btn.setEnabled(bool(detected))

        self.randomize_btn.setEnabled(bool(detected))

        self.model_combo.setEnabled(True)
        self.gain_spin.setEnabled(True)
        self.keep_stems_cb.setEnabled(True)

        # DSP controls
        for w in (self.enable_eq, self.eq_cutoff, self.eq_order,
                  self.enable_deess, self.deess_thresh, self.deess_ratio,
                  self.enable_nr, self.nr_window, self.nr_strength, self.nr_floor):
            w.setEnabled(True)

        for sp in self.offset_spins.values():
            sp.setEnabled(True)

        self.detect_worker = None
        self.render_worker = None


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(820, 780)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
