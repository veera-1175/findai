"""Scan orchestration + upload preview."""

from __future__ import annotations

import base64
import io
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.config import ALL_EXTENSIONS, MAX_FILE_SIZE, UPLOAD_DIR
from app.db import save_scan
from app.engine import detect_content_type, run_analysis
from app.models import AnalysisResult, ContentType


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def build_preview(
    content_type: ContentType,
    filename: str | None,
    file_path: str | None,
    text_input: str | None,
    meta: dict | None = None,
) -> dict:
    meta = meta or {}
    preview = {
        "content_type": content_type.value,
        "filename": filename,
        "label": "Pasted text" if text_input else (filename or content_type.value.title()),
    }
    if text_input:
        preview.update(text=text_input, text_excerpt=text_input[:1200] + ("…" if len(text_input) > 1200 else ""),
                       char_count=len(text_input), word_count=len(text_input.split()))
        return preview
    if not file_path or not Path(file_path).exists():
        return preview
    preview["file_size"] = Path(file_path).stat().st_size
    preview["file_size_label"] = _human_size(preview["file_size"])
    if content_type == ContentType.IMAGE:
        try:
            from PIL import Image
            img = Image.open(file_path).convert("RGB")
            preview["dimensions"] = f"{meta.get('width', img.size[0])} × {meta.get('height', img.size[1])}"
            thumb = img.copy()
            thumb.thumbnail((520, 520))
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=88)
            preview["image_thumb"] = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            preview["preview_note"] = "Image preview unavailable"
    elif content_type == ContentType.VIDEO:
        preview.update(duration=meta.get("duration_seconds"), resolution=meta.get("resolution"))
        try:
            import cv2
            from PIL import Image
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail((520, 320))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                preview["image_thumb"] = base64.b64encode(buf.getvalue()).decode("ascii")
                preview["preview_note"] = "First frame of video"
        except Exception:
            pass
    elif content_type == ContentType.AUDIO:
        preview.update(duration=meta.get("duration_seconds"), sample_rate=meta.get("sample_rate"))
    elif content_type == ContentType.DOCUMENT and meta.get("text_excerpt"):
        preview["text_excerpt"] = meta["text_excerpt"]
        preview["extracted_chars"] = meta.get("extracted_chars", 0)
    return preview


def save_upload(upload: UploadFile, scan_id: str) -> tuple[str, str]:
    if not upload.filename:
        raise HTTPException(400, "No filename provided")
    ext = ("." + upload.filename.rsplit(".", 1)[-1].lower()) if "." in upload.filename else ""
    if ext not in ALL_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    dest = UPLOAD_DIR / f"{scan_id}{ext}"
    size = 0
    with dest.open("wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File too large. Max {MAX_FILE_SIZE // (1024 * 1024)} MB")
            f.write(chunk)
    return str(dest), upload.filename


def perform_scan(file: UploadFile | None = None, text: str = "") -> AnalysisResult:
    scan_id = str(uuid.uuid4())[:8]
    file_path, filename, text_input = None, None, text.strip() or None
    try:
        if file and file.filename:
            file_path, filename = save_upload(file, scan_id)
        content_type = detect_content_type(filename, text_input)
        result = run_analysis(scan_id, content_type, file_path, filename, text_input)
        result.metadata["preview"] = build_preview(content_type, filename, file_path, text_input, result.metadata)
        result.metadata["scanned_at"] = datetime.now(timezone.utc).isoformat()
        result.metadata["engine"] = "findai"
        save_scan(result)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        if file_path:
            Path(file_path).unlink(missing_ok=True)
