"""Multimodal content analyzers — image, video, audio, text, document."""

from __future__ import annotations

import io
import re
import wave
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from PIL import Image, ExifTags

from app.engine.forensics import (
    ai_likelihood_from_entropy,
    analyze_audio_forensics,
    analyze_edits,
    global_ela,
)
from app.engine.fusion import fuse_signals
from app.engine.ml import predict_ai_image, predict_ai_text
from app.models import AnalysisResult, ContentType, Signal

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

AI_PHRASES = [
    r"\bin conclusion\b", r"\bit'?s worth noting\b", r"\bdelve\b", r"\bmoreover\b",
    r"\bfurthermore\b", r"\bas an ai\b", r"\bleverage\b", r"\bseamless(ly)?\b",
    r"\brobust solution\b", r"\bever-evolving\b", r"\bcomprehensive guide\b",
]
AI_WORDS = {
    "delve", "moreover", "furthermore", "comprehensive", "robust", "seamless",
    "leverage", "utilize", "multifaceted", "landscape", "pivotal", "nuanced",
}


def _fail(scan_id: str, ctype: ContentType, filename: str | None, name: str, detail: str, limitations: list[str]) -> AnalysisResult:
    signals = [Signal(name=name, label="Analysis", score=0.5, detail=detail)]
    verdict, confidence, reasons = fuse_signals(signals, content_type=ctype.value)
    return AnalysisResult(scan_id, ctype, filename, verdict, confidence, reasons, signals, limitations)


def _image_meta(img: Image.Image) -> tuple[float, str]:
    try:
        exif = img.getexif()
        if not exif:
            return 0.45, "No EXIF metadata (common in AI images)"
        useful = sum(1 for t, v in exif.items() if ExifTags.TAGS.get(t, t) in ("Make", "Model", "DateTime", "Software") and v)
        return (0.12, f"Camera metadata found ({useful} tags)") if useful else (0.4, "EXIF lacks camera info")
    except Exception:
        return 0.35, "Could not read metadata"


def analyze_image(scan_id: str, file_path: str, filename: str) -> AnalysisResult:
    img = Image.open(file_path).convert("RGB")
    limitations = []
    if max(img.size) < 128:
        limitations.append("Low resolution — use 256px+ for best accuracy.")
    ai_score, ml_detail, ml_meta = predict_ai_image(img)
    signals = [Signal("ml_detector", "Deep learning AI detector", ai_score, ml_detail)]
    for name, label, score, detail in analyze_edits(img):
        signals.append(Signal(name, label, score, detail))
    ms, md = _image_meta(img)
    es, ed = global_ela(img)
    signals += [Signal("metadata", "EXIF metadata", ms, md), Signal("ela", "Global compression", es, ed)]
    if ml_meta.get("error"):
        limitations.append("ML model not loaded — pip install torch transformers")
    verdict, confidence, reasons = fuse_signals(signals, "image")
    return AnalysisResult(scan_id, ContentType.IMAGE, filename, verdict, confidence, reasons, signals, limitations,
                          {"width": img.size[0], "height": img.size[1], "format": Path(filename).suffix.lstrip("."), "ml": ml_meta})


def analyze_video(scan_id: str, file_path: str, filename: str) -> AnalysisResult:
    if not HAS_CV2:
        return _fail(scan_id, ContentType.VIDEO, filename, "opencv", "OpenCV unavailable", ["Install opencv-python-headless"])
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return _fail(scan_id, ContentType.VIDEO, filename, "decode", "Failed to open video", ["Could not read video"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frames = []
    if total > 0:
        for idx in np.linspace(0, total - 1, min(6, total), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    else:
        while len(frames) < 6:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    cap.release()
    if not frames:
        return _fail(scan_id, ContentType.VIDEO, filename, "decode", "No frames extracted", ["Empty video"])

    ml_scores, edit_scores = [], []
    for frame in frames[:5]:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ml_scores.append(predict_ai_image(pil)[0])
        edit_scores.extend(s[2] for s in analyze_edits(pil))
    signals = []
    if ml_scores:
        avg, peak = sum(ml_scores) / len(ml_scores), max(ml_scores)
        combined = 0.6 * peak + 0.4 * avg
        signals.append(Signal("ml_frame_avg", "Deep learning (frames)", combined,
                              f"AI across {len(ml_scores)} frames: avg {avg:.0%}, peak {peak:.0%}"))
    if edit_scores and max(edit_scores) >= 0.4:
        signals.append(Signal("block_ela", "Frame edit forensics", max(edit_scores),
                              f"Peak edit signal {max(edit_scores):.0%}"))
    diffs = []
    for i in range(1, min(len(frames), 6)):
        a = cv2.resize(cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY).astype(np.float32), (64, 64))
        b = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32), (64, 64))
        diffs.append(float(np.mean(np.abs(a - b))))
    std_d = float(np.std(diffs)) if diffs else 0
    mean_d = float(np.mean(diffs)) if diffs else 0
    temp_s = 0.62 if diffs and std_d > mean_d * 0.8 and mean_d > 5 else 0.18
    signals.append(Signal("temporal", "Temporal consistency", temp_s,
                          f"Irregular frame changes (σ={std_d:.1f})" if temp_s > 0.5 else "Frame transitions consistent"))
    duration = total / fps if fps and total else 0
    verdict, confidence, reasons = fuse_signals(signals, "video")
    return AnalysisResult(scan_id, ContentType.VIDEO, filename, verdict, confidence, reasons, signals,
                          [f"Analyzed {len(frames)} frames with ML + forensics."],
                          {"fps": round(fps, 2), "frames_analyzed": len(frames), "total_frames": total,
                           "duration_seconds": round(duration, 2),
                           "resolution": f"{frames[0].shape[1]}x{frames[0].shape[0]}"})


def _load_audio(file_path: str) -> tuple[np.ndarray, int] | None:
    ext = Path(file_path).suffix.lower()
    if ext == ".wav":
        try:
            with wave.open(file_path, "rb") as wf:
                raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
                if wf.getnchannels() > 1:
                    raw = raw.reshape(-1, wf.getnchannels()).mean(axis=1)
                peak = np.max(np.abs(raw)) or 1.0
                return raw / peak, wf.getframerate()
        except Exception:
            pass
    if HAS_PYDUB and ext in {".mp3", ".m4a", ".ogg", ".flac"}:
        try:
            audio = AudioSegment.from_file(file_path).set_channels(1).set_frame_rate(22050)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            peak = np.max(np.abs(samples)) or 1.0
            return samples / peak, audio.frame_rate
        except Exception:
            pass
    return None


def analyze_audio(scan_id: str, file_path: str, filename: str) -> AnalysisResult:
    loaded = _load_audio(file_path)
    if loaded is None:
        return _fail(scan_id, ContentType.AUDIO, filename, "decode", "Unable to decode audio",
                     ["Install ffmpeg for MP3/M4A support"])
    samples, sr = loaded
    duration = len(samples) / sr
    signals = [Signal(n, l, s, d) for n, l, s, d in analyze_audio_forensics(samples, sr)]
    chunk = samples[: min(len(samples), sr * 3)]
    fft = np.abs(np.fft.rfft(chunk))
    fft = fft[fft > 0]
    if len(fft) > 10:
        flat = float(np.exp(np.mean(np.log(fft + 1e-10))) / (np.mean(fft) + 1e-10))
        sp = min(0.85, 0.45 + flat * 1.5) if flat > 0.12 else 0.15
        signals.append(Signal("spectral", "Spectral analysis", sp, f"Flatness {flat:.3f}"))
    verdict, confidence, reasons = fuse_signals(signals, "audio")
    return AnalysisResult(scan_id, ContentType.AUDIO, filename, verdict, confidence, reasons, signals,
                          ["Librosa pitch/MFCC + splice detection (local)."],
                          {"duration_seconds": round(duration, 2), "sample_rate": sr})


def analyze_text(scan_id: str, text: str, filename: str | None = None) -> AnalysisResult:
    text = text.strip()
    signals, limitations = [], []
    if len(text) < 50:
        limitations.append("Use 50+ characters for best neural accuracy.")
    ml_score, ml_detail, ml_meta = predict_ai_text(text)
    signals.append(Signal("ml_text_detector", "RoBERTa AI detector", ml_score, ml_detail))
    sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if len(p.strip()) > 3]
    word_count = len(text.split())
    if len(sentences) >= 5 and word_count >= 80:
        lens = [len(s.split()) for s in sentences]
        ratio = (pstdev(lens) / mean(lens)) if mean(lens) else 0
        bs = 0.75 if ratio < 0.25 else 0.5 if ratio < 0.4 else 0.2
        signals.append(Signal("burstiness", "Sentence variety", bs, f"Variation ratio {ratio:.2f}"))
    hits = sum(1 for p in AI_PHRASES if re.search(p, text, re.I))
    if hits:
        signals.append(Signal("ai_phrases", "AI phrase patterns", min(1.0, hits * 0.2), f"{hits} AI pattern(s)"))
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    ai_hits = len(words & AI_WORDS)
    if ai_hits:
        signals.append(Signal("ai_vocabulary", "AI vocabulary", min(1.0, ai_hits * 0.15), f"{ai_hits} AI word(s)"))
    personal = len(re.findall(r"\b(I|me|my|we|our|I'm|I've|lol|haha)\b", text, re.I))
    if personal >= 2:
        signals.append(Signal("human_markers", "Human markers", 0.12, f"{personal} personal markers"))
    ent_s, ent_d = ai_likelihood_from_entropy(text)
    if ent_s >= 0.35 and word_count >= 100:
        signals.append(Signal("entropy", "Statistical entropy", ent_s, ent_d))
    verdict, confidence, reasons = fuse_signals(signals, "text")
    return AnalysisResult(scan_id, ContentType.TEXT, filename, verdict, confidence, reasons, signals, limitations,
                          {"char_count": len(text), "word_count": word_count, "sentences": len(sentences), "ml": ml_meta})


def analyze_document(scan_id: str, file_path: str, filename: str) -> AnalysisResult:
    ext = Path(filename).suffix.lower()
    text, meta, signals, limitations = "", {}, [], []
    if ext == ".pdf" and HAS_PYPDF:
        reader = PdfReader(file_path)
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        if reader.metadata:
            meta = {k: str(reader.metadata.get(k, "")) for k in ("/Creator", "/Producer", "/Title")}
            meta = {"creator": meta.get("/Creator", ""), "producer": meta.get("/Producer", ""), "title": meta.get("/Title", "")}
        for tool in ("chatgpt", "openai", "claude", "gemini", "copilot"):
            if tool in meta.get("producer", "").lower() or tool in meta.get("creator", "").lower():
                signals.append(Signal("pdf_creator", "Document creator", 0.8, f"Metadata references '{tool}'"))
                break
    elif ext == ".docx" and HAS_DOCX:
        text = "\n".join(p.text for p in DocxDocument(file_path).paragraphs if p.text.strip())
    elif ext == ".txt":
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    if text.strip():
        tr = analyze_text(scan_id + "_t", text, filename)
        limitations.extend(tr.limitations)
        signals.extend(Signal(f"text_{s.name}", f"Text: {s.label}", s.score, s.detail) for s in tr.signals)
    else:
        signals.append(Signal("no_text", "Text extraction", 0.45, "Could not extract text"))
        limitations.append("No extractable text — may be scanned image PDF.")
    verdict, confidence, reasons = fuse_signals(signals, "document")
    excerpt = (text[:1200] + "…") if len(text) > 1200 else text
    return AnalysisResult(scan_id, ContentType.DOCUMENT, filename, verdict, confidence, reasons, signals, limitations,
                          {"extracted_chars": len(text), "pdf_meta": meta, "text_excerpt": excerpt})
