# FindAI

**Free multimodal AI & fake content detector** — runs in the browser, locally, or on [Render](https://render.com) cloud. No paid APIs.

Detect whether images, videos, audio, text, or documents are likely **real**, **AI-generated**, or **AI-edited**.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)
![Docker](https://img.shields.io/badge/Deploy-Render-blue)
![License](https://img.shields.io/badge/Cost-Free-brightgreen)

## Features

| Module | What it does |
|--------|----------------|
| **Image** | ViT neural network + ELA edit forensics |
| **Video** | Frame-by-frame ML + temporal consistency |
| **Audio** | Synthetic voice + splice detection |
| **Text** | RoBERTa detector + entropy heuristics |
| **Documents** | PDF metadata + text pipeline |
| **Dashboard** | Verdict analytics + engine status |
| **REST API** | JSON analyze, export, stats, health |
| **History** | SQLite scan history |

## Deploy to Render (cloud — no local setup)

Everything runs on Render's servers. Models download from Hugging Face on first scan (~850 MB total, free).

### One-click deploy

1. Push this repo to **GitHub** (repo name e.g. `findai`)
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect the repo — Render reads `render.yaml` automatically
4. Click **Apply** — build takes ~10–15 minutes (PyTorch + deps)

Your app will be live at `https://findai.onrender.com` (or similar).

### Manual deploy (Docker)

1. **New Web Service** → connect GitHub repo
2. **Runtime:** Docker
3. **Dockerfile path:** `./Dockerfile`
4. **Plan:** Free (see RAM note below)
5. Deploy

### Cloud environment variables (set automatically by `render.yaml`)

| Variable | Purpose |
|----------|---------|
| `RENDER=true` | Enables cloud mode |
| `DATA_DIR=/tmp/findai` | Writable uploads + SQLite |
| `PRELOAD_ML=false` | Lazy-load models (saves RAM at startup) |
| `HF_HOME=/tmp/huggingface` | Model cache directory |

### Render free tier — important

| | Free | Standard ($25/mo) |
|--|------|-------------------|
| RAM | 512 MB | 2 GB |
| ML models | May fail / slow cold start | Recommended for full ML |
| Sleep | Spins down after 15 min idle | Always on |
| History DB | Resets on redeploy | Use persistent disk |

**Free tier:** UI, API, text heuristics, and demos work. Image/text **neural models need ~1 GB RAM** — if scans fail with memory errors, upgrade to **Standard (2 GB)** on Render or use local setup.

First scan after sleep can take **2–5 minutes** (wake + model download).

## Run locally (Windows)

```bat
setup.bat
start.bat
```

Open **http://localhost:8000**

## Run locally (manual)

```bash
cd findai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## Try demos

On the home page, click **AI-style text** or **Human-style text** — instant demo without uploading.

## API

```bash
# Health check
curl https://YOUR-APP.onrender.com/api/health

# Analyze text
curl -X POST https://YOUR-APP.onrender.com/api/analyze \
  -F "text=Moreover it is worth noting that we must delve into comprehensive solutions."

# Stats
curl https://YOUR-APP.onrender.com/api/stats
```

Swagger docs: `/docs`

## Run tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Project structure

```
findai/
├── app/
│   ├── main.py           # FastAPI routes
│   ├── config.py         # Local + cloud settings
│   ├── db.py             # SQLite + analytics
│   ├── models.py
│   ├── demo.py
│   ├── core/
│   │   └── scan.py       # Upload + analyze pipeline
│   ├── engine/
│   │   ├── ml.py         # Hugging Face models
│   │   ├── forensics.py  # ELA, entropy, audio
│   │   ├── analyze.py    # All modality analyzers
│   │   └── fusion.py     # Verdict engine
│   ├── templates/        # Jinja2 UI
│   └── static/           # CSS + JS
├── tests/
├── Dockerfile            # Render / Docker deploy
├── render.yaml           # Render Blueprint
├── requirements.txt
├── run.py
└── PROJECT_DOCUMENTATION.md
```

## Tech stack (all free)

FastAPI · Uvicorn · Jinja2 · PyTorch (CPU) · Transformers · Pillow · OpenCV · Librosa · SciPy · SQLite

## Academic documentation

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for MCA/project report material.

## Disclaimer

FindAI gives **probability estimates** for education and awareness — not legal proof. Accuracy varies with compression quality and new AI models.
