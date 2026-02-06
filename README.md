# Face Recognition System (ArcFace ONNX + 5-Point Alignment)

A lightweight, **CPU-friendly face recognition** project that runs in real time with a webcam. It detects faces with **Haar cascades**, extracts **5 keypoints** via **MediaPipe FaceMesh**, aligns faces to **112×112** using the ArcFace template, then generates **ArcFace embeddings** with an **ONNX** model for matching.

This repo is intended for **learning, experimentation, and desktop/laptop demos** (works well on macOS, including Apple Silicon).

## What’s inside

- **Enrollment**: capture multiple aligned samples per person and build a small embedding database.
- **Recognition**: detect multiple faces, embed each, and match against the database with a cosine-distance threshold.
- **Evaluation**: offline threshold tuning using saved enrollment crops.
- **Modular pipeline** under `src/` (detector, alignment, embedder, tools).

## Requirements

- Python 3.9+ recommended
- A webcam (for `enroll` / `recognize`)

Install dependencies:

```bash
pip install -r requirements.txt
```

If `mediapipe` fails to import on your machine, try a known working version:

```bash
pip install "mediapipe==0.10.21"
```

## Quickstart

Create the expected folders (`data/enroll`, `data/db`, etc.):

```bash
python init_project.py
```

### 1) Enroll (register) a person

```bash
python -m src.enroll
```

- You’ll be prompted for a name (e.g., `Alice`).
- Aligned crops can be saved under `data/enroll/<name>/`.
- A database is written to:
  - `data/db/face_db.npz` (name → embedding vector)
  - `data/db/face_db.json` (metadata)

Controls (during enrollment):

- `SPACE`: capture one sample (if a face is found)
- `a`: toggle auto-capture
- `s`: save enrollment to DB
- `r`: reset NEW samples (keeps existing crops on disk)
- `q`: quit

### 2) Recognize (identify) faces

```bash
python -m src.recognize
```

The recognizer loads `data/db/face_db.npz` and labels faces as the closest known identity or `Unknown`.

Controls (during recognition):

- `q`: quit
- `r`: reload DB from disk
- `+` / `-`: adjust the matching threshold (cosine distance) live
- `d`: toggle debug overlay

### 3) Evaluate / tune the threshold (optional)

This uses your saved aligned enrollment crops under `data/enroll/<name>/*.jpg` to estimate genuine vs impostor distance distributions and suggest a threshold.

```bash
python -m src.evaluate
```

## Data & outputs

- **Enrollment crops** (optional): `data/enroll/<person_name>/*.jpg` (aligned `112×112`)
- **Database**: `data/db/face_db.npz` (used by recognition)

## Project layout

```
models/
  embedder_arcface.onnx
data/
  enroll/              # aligned crops per identity (optional)
  db/                  # face_db.npz + metadata
src/
  enroll.py            # build/update DB
  recognize.py         # real-time multi-face recognition
  evaluate.py          # offline threshold tuning
  haar_5pt.py          # Haar + FaceMesh → 5-point keypoints + alignment
  embed.py             # ArcFace ONNX embedder demo
init_project.py
requirements.txt
```

## Notes / disclaimers

- This is a **CPU-only** demo pipeline (no GPU required).
- Designed for **small-scale** recognition (a handful to dozens of identities).
- Not production-hardened (spoofing, privacy, security, and dataset bias are out of scope).
