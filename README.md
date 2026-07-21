# FindAI

### Free multimodal AI-content detector for text, images, and video frames

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-46E3B7?logo=render&logoColor=white)](https://findai-detector.onrender.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Paste text, drop an image, or upload a video — get an explainable authenticity score without a paywall.

Built by **[Veera](https://github.com/veera-1175)**.

**Live demo:** [findai-detector.onrender.com](https://findai-detector.onrender.com)

---

## What it does

FindAI combines **fast heuristics** with **optional neural models** so the product works on a free hosting tier and upgrades cleanly when you have more RAM:

| Modality | Approach |
|----------|----------|
| **Text** | Style / predictability signals + optional transformer scoring |
| **Images** | Visual artifact / model-based checks (when models load) |
| **Video** | Frame sampling → same image pipeline |
| **Demos** | One-click AI-style vs human-style samples — no upload needed |

Results include a **score**, **label**, and **human-readable explanation** so users understand *why*, not just a black-box percentage.

---

## Highlights

- Zero sign-up web UI — paste, upload, scan
- REST API for scripting (`/api/analyze`, `/api/health`)
- Scan history (SQLite; ephemeral on free Render redeploys)
- Graceful degradation when ML models can't load (heuristics still work)
- One-command local setup on Windows (`setup.bat` / `start.bat`)

---

## Table of contents

1. [Live demo](#live-demo)
2. [Quick start](#quick-start)
3. [API](#api)
4. [Deploy (Render)](#deploy-render)
5. [Project layout](#project-layout)
6. [Interview walkthrough](#interview-walkthrough)
7. [License](#license)

---

## Live demo

| | |
|--|--|
| **URL** | [https://findai-detector.onrender.com](https://findai-detector.onrender.com) |
| **Cold start** | Free tier may sleep — first request can take 1–2 minutes |
| **ML note** | Neural models need ~1 GB RAM; upgrade to Standard if scans OOM |

Try the **AI-style text** / **Human-style text** buttons on the home page for an instant demo.

---

## Quick start

```bat
git clone https://github.com/veera-1175/findai.git
cd findai
setup.bat
start.bat
```

Open **http://localhost:8000**

Manual:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

---

## API

```bash
curl https://YOUR-APP.onrender.com/api/health

curl -X POST https://YOUR-APP.onrender.com/api/analyze \
  -F "text=Moreover it is worth noting that we must delve into comprehensive solutions."
```

Multipart fields support `text`, `image`, and `video` depending on the request.

---

## Deploy (Render)

| Setting | Free | Paid / persistent |
|---------|------|-------------------|
| App | Web service from this repo | Same |
| ML models | May fail / slow cold start | Recommended (~2 GB) |
| Sleep | Spins down after idle | Always on |
| History DB | Resets on redeploy | Persistent disk |

`render.yaml` (if present) or manual: Python runtime, start `python run.py`, expose the configured port.

---

## Project layout

```
findai/
├── app/              # FastAPI routes, scoring, static UI
├── models/           # Optional weights / loaders
├── run.py            # Entry
├── requirements.txt
├── setup.bat / start.bat
└── README.md
```

---

## Interview walkthrough

| Topic | Talking point |
|-------|----------------|
| Product sense | Free multimodal detector UX, not a notebook dump |
| Engineering | FastAPI async API + static frontend, file uploads, health checks |
| Reliability | Heuristics when models can't load; clear UX on cold starts |
| Ops | Render deploy, RAM constraints, cold start awareness |

**60-second demo:** open live site → click demo text → show score + explanation → optional image upload.

---

## License

MIT © [Veera](https://github.com/veera-1175)

**FindAI** — authenticity checks without the paywall.
