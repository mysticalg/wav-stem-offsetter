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
    QProgressBar, QTextEdit, QComboBox, QCheckBox
)


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

    # Stream output to log
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
    Expects demucs outputs like drums.wav, bass.wav, other.wav, vocals.wav
    """
    stem_files = {p.stem.lower(): p for p in stems_dir.glob("*.wav")}
    if not stem_files:
        raise FileNotFoundError(f"No stem wavs found in {stems_dir}")

    stems_audio = {}
    sr = None
    max_len = 0

    for name, path in stem_files.items():
        audio, sr_i = sf.read(path, always_2d=True)  # (n, ch)
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


def mix_and_write(stems_audio: dict, sr: int, offsets_ms: dict, master_gain: float, out_wav: Path):
    """
    Applies offsets to each stem, sums, normalizes to avoid clipping, applies gain, writes out_wav.
    offsets_ms: {stem_name: integer ms} (may be 0..200; negatives not used by GUI but supported)
    """
    # convert ms -> samples
    offsets_samples = {
        name: int(round((ms / 1000.0) * sr))
        for name, ms in offsets_ms.items()
    }

    # apply shifts
    shifted = {}
    for name, audio in stems_audio.items():
        shifted[name] = apply_offset(audio, offsets_samples.get(name, 0))

    # mix
    mix = np.zeros_like(next(iter(shifted.values())))
    for audio in shifted.values():
        mix += audio

    # basic anti-clip normalize (only if needed), then gain
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        mix = mix / peak

    mix *= float(master_gain)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, mix, sr)


# --------------------------
# Worker thread
# --------------------------
class ProcessWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, input_wav: Path, out_wav: Path, model: str, offsets_ms: dict,
                 master_gain: float, keep_stems: bool):
        super().__init__()
        self.input_wav = input_wav
        self.out_wav = out_wav
        self.model = model
        self.offsets_ms = offsets_ms
        self.master_gain = master_gain
        self.keep_stems = keep_stems

    def run(self):
        temp_root = None
        try:
            self.progress.emit(5)
            self.log.emit(f"Input: {self.input_wav}")
            self.log.emit(f"Output: {self.out_wav}")
            self.log.emit(f"Model: {self.model}")
            self.log.emit(f"Offsets (ms): {self.offsets_ms}")
            self.log.emit(f"Master gain: {self.master_gain}")

            temp_root = Path(tempfile.mkdtemp(prefix="stem_offset_gui_"))
            demucs_out = temp_root / "demucs_out"

            self.progress.emit(10)
            stems_dir = run_demucs(self.input_wav, demucs_out, self.model, log_cb=self.log.emit)

            self.progress.emit(65)
            self.log.emit(f"Loading stems from: {stems_dir}")
            stems_audio, sr = load_stems(stems_dir)

            # Keep only offsets for stems present
            filtered_offsets = {k: self.offsets_ms.get(k, 0) for k in stems_audio.keys()}
            self.log.emit(f"Detected stems: {list(stems_audio.keys())}")
            self.log.emit(f"Using offsets (ms): {filtered_offsets}")

            self.progress.emit(80)
            mix_and_write(stems_audio, sr, filtered_offsets, self.master_gain, self.out_wav)

            self.progress.emit(100)
            self.done.emit(f"Saved: {self.out_wav}")

        except Exception as e:
            self.failed.emit(str(e))
        finally:
            if temp_root and temp_root.exists() and not self.keep_stems:
                try:
                    shutil.rmtree(temp_root, ignore_errors=True)
                except Exception:
                    pass


# --------------------------
# Main Window
# --------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stem Offset Remixer (Demucs)")

        self.setAcceptDrops(True)

        self.current_wav: Path | None = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.drop_label = QLabel("Drag & drop a WAV file here\n(or click “Open WAV…”)")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet(
            "QLabel { border: 2px dashed #777; padding: 24px; font-size: 14px; }"
        )
        layout.addWidget(self.drop_label)

        btn_row = QHBoxLayout()
        self.open_btn = QPushButton("Open WAV…")
        self.save_btn = QPushButton("Save Processed WAV As…")
        self.save_btn.setEnabled(False)
        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)

        # Options group
        opt_box = QGroupBox("Options")
        opt_layout = QFormLayout(opt_box)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["htdemucs", "htdemucs_ft", "mdx_extra", "mdx"])
        self.model_combo.setCurrentText("htdemucs")
        opt_layout.addRow("Demucs model", self.model_combo)

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 1.0)
        self.gain_spin.setSingleStep(0.05)
        self.gain_spin.setValue(0.95)
        opt_layout.addRow("Master gain", self.gain_spin)

        self.keep_stems_cb = QCheckBox("Keep temporary Demucs files (debug)")
        self.keep_stems_cb.setChecked(False)
        opt_layout.addRow("", self.keep_stems_cb)

        layout.addWidget(opt_box)

        # Offsets group
        offsets_box = QGroupBox("Offsets per stem (ms)")
        offsets_layout = QFormLayout(offsets_box)

        # Common demucs stems; if a model outputs differently, we’ll ignore missing ones
        self.offset_spins = {}
        for stem in ["drums", "bass", "other", "vocals"]:
            sp = QSpinBox()
            sp.setRange(0, 200)  # GUI requested 1..200, but 0 is useful too; you can ignore 0 if you want.
            sp.setValue(10)
            self.offset_spins[stem] = sp
            offsets_layout.addRow(stem.capitalize(), sp)

        rand_row = QHBoxLayout()
        self.rand_min = QSpinBox()
        self.rand_min.setRange(1, 200)
        self.rand_min.setValue(1)
        self.rand_max = QSpinBox()
        self.rand_max.setRange(1, 200)
        self.rand_max.setValue(200)
        self.randomize_btn = QPushButton("Randomize")
        rand_row.addWidget(QLabel("Random range"))
        rand_row.addWidget(self.rand_min)
        rand_row.addWidget(QLabel("to"))
        rand_row.addWidget(self.rand_max)
        rand_row.addWidget(QLabel("ms"))
        rand_row.addWidget(self.randomize_btn)
        offsets_layout.addRow(rand_row)

        layout.addWidget(offsets_box)

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
        self.save_btn.clicked.connect(self.save_processed)
        self.randomize_btn.clicked.connect(self.randomize_offsets)

        self.worker: ProcessWorker | None = None

    # Drag & drop
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

    def log(self, text: str):
        self.log_box.append(text)

    def set_current_wav(self, path: Path):
        self.current_wav = path
        self.drop_label.setText(f"Loaded:\n{path}")
        self.save_btn.setEnabled(True)
        self.progress.setValue(0)
        self.log_box.clear()
        self.log(f"Loaded WAV: {path}")

    def open_wav(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open WAV", "", "WAV files (*.wav)"
        )
        if file_path:
            self.set_current_wav(Path(file_path))

    def randomize_offsets(self):
        lo = self.rand_min.value()
        hi = self.rand_max.value()
        if lo > hi:
            lo, hi = hi, lo
        for stem, sp in self.offset_spins.items():
            sp.setValue(random.randint(lo, hi))
        self.log(f"Randomized offsets in range {lo}..{hi} ms")

    def gather_offsets(self) -> dict:
        return {stem: sp.value() for stem, sp in self.offset_spins.items()}

    def save_processed(self):
        if not self.current_wav or not self.current_wav.exists():
            QMessageBox.warning(self, "No input", "Please load a WAV first.")
            return

        default_name = self.current_wav.with_name(self.current_wav.stem + "_shifted.wav")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed WAV As", str(default_name), "WAV files (*.wav)"
        )
        if not out_path:
            return

        # Start worker
        self.save_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.randomize_btn.setEnabled(False)
        for sp in self.offset_spins.values():
            sp.setEnabled(False)

        self.progress.setValue(0)
        self.log("Starting processing... (this can take a bit on the first run)")

        self.worker = ProcessWorker(
            input_wav=self.current_wav,
            out_wav=Path(out_path),
            model=self.model_combo.currentText(),
            offsets_ms=self.gather_offsets(),
            master_gain=self.gain_spin.value(),
            keep_stems=self.keep_stems_cb.isChecked(),
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_done(self, msg: str):
        self.log(msg)
        QMessageBox.information(self, "Done", msg)
        self._unlock_ui()

    def on_failed(self, err: str):
        self.log("ERROR: " + err)
        QMessageBox.critical(self, "Error", err)
        self._unlock_ui()

    def _unlock_ui(self):
        self.save_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self.randomize_btn.setEnabled(True)
        for sp in self.offset_spins.values():
            sp.setEnabled(True)
        self.worker = None


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(720, 650)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
