import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root (code, templates, static)
BASE_DIR = Path(__file__).resolve().parent.parent

# Writable data dir — /tmp on Render, project root locally
IS_CLOUD = os.getenv("RENDER") == "true" or os.getenv("CLOUD", "").lower() in ("1", "true", "yes")
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/findai" if IS_CLOUD else BASE_DIR))
DATA_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false" if IS_CLOUD else "true").lower() == "true"
PRELOAD_ML = os.getenv("PRELOAD_ML", "false" if IS_CLOUD else "true").lower() == "true"

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024 if IS_CLOUD else 100 * 1024 * 1024)))

ALLOWED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"},
    "video": {".mp4", ".mov", ".avi", ".webm", ".mkv"},
    "audio": {".mp3", ".wav", ".m4a", ".ogg", ".flac"},
    "document": {".pdf", ".docx", ".txt"},
    "text": set(),
}

ALL_EXTENSIONS = set().union(*ALLOWED_EXTENSIONS.values())
