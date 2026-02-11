📄 README.md
# WAV Stem Offsetter (Windows GUI)

A simple **Windows GUI** tool that:

- extracts **stems** from a WAV file (via Demucs)
- applies **per-stem time offsets**
- recombines stems into a new WAV

Designed for fast experimentation with phase shifts, groove offsets, and creative remix processing.

---

## ✨ Features

- Drag & drop WAV files
- Automatic stem detection
- Manual per-stem offset control (0–200ms)
- Randomize offsets (optionally unique)
- Supports tracks with few or many stems
- Render and save reconstructed WAV

---

## 🧰 Requirements

### System
- Windows 10 / 11

### Software
- Python **3.12** (recommended)

Download:
https://www.python.org/downloads/windows/

During install:


✔ Add Python to PATH


---

## ⚙️ Installation

Open PowerShell inside the project folder.

### 1️⃣ Create virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip setuptools wheel

2️⃣ Install dependencies

Install PyTorch FIRST (CPU version works everywhere):

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu


Then install remaining packages:

pip install PySide6 demucs soundfile numpy

3️⃣ (Optional but Recommended) Install FFmpeg

Some audio operations rely on FFmpeg.

winget install Gyan.FFmpeg


Restart PowerShell afterwards.

▶️ Running the App
.\.venv\Scripts\activate
python StemOffsetGUI.py

🧭 Usage Guide

Launch the application

Drag & drop a WAV file

Click Extract Stems / Detect

Adjust offsets per stem or click Randomize

Click Render + Save As…

Choose output filename

Each stem receives its own timing offset before reconstruction.

🛠 Troubleshooting
Demucs TorchAudio / TorchCodec Errors

Newer TorchAudio versions sometimes require TorchCodec.

Possible fixes:

Install torchcodec and FFmpeg shared builds

OR downgrade Torch/TorchAudio to avoid TorchCodec requirement

NumPy Import Errors

If you see errors about incompatible compiled modules:

Importing the numpy C-extensions failed


Fix:

deactivate
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\activate


Reinstall dependencies.

Large Temporary Files

Demucs generates large temporary stem files during processing.

These are ignored by .gitignore.

📁 Project Structure
StemOffsetGUI.py   # Main application
README.md          # Documentation
.gitignore
