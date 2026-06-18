# FindAI — Project Documentation

## Project Title
**FindAI: A Free Multimodal AI & Fake Content Authenticity Detector (Cloud-Ready)**

## Abstract
FindAI is a web application that detects whether digital content (images, videos, audio, text, documents) is likely authentic, AI-generated, or AI-edited. It uses **100% free open-source** Python libraries and Hugging Face models — no paid APIs. The system can run **locally** or deploy to **Render cloud** via Docker with no local dependencies for end users.

## Problem Statement
AI-generated media and text are increasingly hard to distinguish from authentic content. Commercial tools often require subscriptions. FindAI provides a unified, free, deployable solution for education and awareness.

## Objectives
1. Multimodal detection in one platform (image, video, audio, text, document)
2. Ensemble of neural networks + digital forensics
3. Web UI with dashboard, history, and export
4. REST API for programmatic access
5. Cloud deployment on Render without local setup for users

## System Architecture

```
User (Browser)
      │
      ▼
FastAPI (Render / Local)
      │
      ├── Jinja2 UI (Check, Dashboard, History, About)
      ├── REST API (/api/analyze, /api/stats, /api/health)
      │
      ▼
app/core/scan.py — upload routing
      │
      ▼
app/engine/ — detection pipeline
      ├── ml.py         → Hugging Face ViT + RoBERTa
      ├── forensics.py  → ELA, entropy, audio splice
      ├── analyze.py    → per-modality analyzers
      └── fusion.py     → verdict + confidence
      │
      ▼
SQLite (findai.db) + scan preview metadata
```

## Modules

| Module | File | Methods |
|--------|------|---------|
| Image ML | `engine/ml.py` | Organika/sdxl-detector ViT |
| Text ML | `engine/ml.py` | roberta-base-openai-detector |
| Edit forensics | `engine/forensics.py` | Block ELA, noise analysis |
| Video | `engine/analyze.py` | Frame sampling + temporal checks |
| Audio | `engine/forensics.py` | Pitch, spectral flatness, splice |
| Text heuristics | `engine/analyze.py` | Phrases, burstiness, entropy |
| Documents | `engine/analyze.py` | PDF metadata + text pipeline |
| Fusion | `engine/fusion.py` | 4 verdicts with confidence |

## Verdicts
- **Authentic** — no strong AI/manipulation signals
- **Uncertain** — mixed or insufficient evidence
- **AI-Generated** — likely fully AI-produced
- **AI-Edited** — real content with localized manipulation

## Technologies

| Layer | Stack |
|-------|-------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML | PyTorch (CPU), Transformers, Hugging Face |
| Forensics | Pillow, OpenCV, Librosa, SciPy |
| Frontend | Jinja2, HTML/CSS/JS |
| Database | SQLite |
| Cloud | Docker, Render Blueprint |

## Deployment

### Local
```bat
setup.bat && start.bat
```

### Render (cloud)
1. Push to GitHub
2. Render → New Blueprint → select repo
3. `render.yaml` configures Docker web service

Environment: `RENDER=true`, `DATA_DIR=/tmp/findai`, lazy ML loading.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Upload UI |
| GET | `/dashboard` | Analytics |
| POST | `/analyze` | Web scan |
| POST | `/api/analyze` | JSON scan |
| GET | `/api/health` | Status + models |
| GET | `/api/stats` | Scan analytics |
| GET | `/result/{id}/export` | JSON export |

## Limitations
- Probabilistic results, not forensic proof
- Render free tier (512 MB RAM) may limit full ML — Standard plan recommended
- Ephemeral storage on free cloud — history lost on redeploy
- Compressed social media files reduce accuracy

## Conclusion
FindAI demonstrates that multimodal AI detection can be built as a student project using entirely free tools, with optional cloud deployment so users need no local installation.
