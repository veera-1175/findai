"""Local Hugging Face models — image + text detection."""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

IMAGE_MODEL = "Organika/sdxl-detector"
TEXT_MODEL = "roberta-base-openai-detector"
TEXT_CHUNK = 450

_slots: dict[str, dict[str, Any]] = {
    "image": {"model": None, "processor": None, "error": None, "lock": threading.Lock()},
    "text": {"model": None, "processor": None, "error": None, "lock": threading.Lock()},
}


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_image():
    slot = _slots["image"]
    if slot["model"]:
        return
    with slot["lock"]:
        if slot["model"]:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        logger.info("Loading image model: %s", IMAGE_MODEL)
        slot["processor"] = AutoImageProcessor.from_pretrained(IMAGE_MODEL)
        slot["model"] = AutoModelForImageClassification.from_pretrained(IMAGE_MODEL)
        slot["model"].eval().to(_device())
        slot["error"] = None


def _load_text():
    slot = _slots["text"]
    if slot["model"]:
        return
    with slot["lock"]:
        if slot["model"]:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        logger.info("Loading text model: %s", TEXT_MODEL)
        slot["processor"] = AutoTokenizer.from_pretrained(TEXT_MODEL)
        slot["model"] = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL)
        slot["model"].eval().to(_device())
        slot["error"] = None


def _softmax_probs(logits, model):
    import torch
    probs = torch.softmax(logits, dim=-1)[0]
    labels = model.config.id2label
    return {str(labels.get(i, i)).lower(): float(probs[i]) for i in range(len(probs))}


def _ai_from_probs(probs: dict[str, float], ai_keys: tuple[str, ...], real_keys: tuple[str, ...]) -> tuple[float, str]:
    ai_prob = max((p for k, p in probs.items() if any(x in k for x in ai_keys)), default=0.0)
    real_prob = max((p for k, p in probs.items() if any(x in k for x in real_keys)), default=0.0)
    if ai_prob == 0 and real_prob == 0 and len(probs) == 2:
        vals = list(probs.values())
        ai_prob, real_prob = vals[0], vals[1]
    return ai_prob, "AI" if ai_prob >= real_prob else "real"


def model_status(kind: str) -> dict[str, Any]:
    from app.config import ML_ENABLED
    slot = _slots[kind]
    model_id = IMAGE_MODEL if kind == "image" else TEXT_MODEL
    return {
        "model_id": model_id,
        "loaded": slot["model"] is not None,
        "enabled": ML_ENABLED,
        "error": slot["error"],
        "provider": "Hugging Face (free, local)" if ML_ENABLED else "Disabled on cloud (heuristics only)",
    }


def all_model_status() -> dict[str, Any]:
    return {
        "image": model_status("image"),
        "text": model_status("text"),
        "audio": {"engine": "librosa forensics", "provider": "local", "loaded": True},
        "video": {"engine": "OpenCV + image ML", "provider": "local", "loaded": True},
        "document": {"engine": "PDF forensics + text ML", "provider": "local", "loaded": True},
    }


def preload_all() -> dict[str, bool]:
    results = {}
    for kind, loader in (("image", _load_image), ("text", _load_text)):
        try:
            loader()
            results[kind] = True
        except Exception as exc:
            _slots[kind]["error"] = str(exc)
            results[kind] = False
    logger.info("Model preload: %s", results)
    return results


def predict_ai_image(img) -> tuple[float, str, dict[str, Any]]:
    from app.config import ML_ENABLED
    if not ML_ENABLED:
        return 0.45, "Neural model disabled in cloud mode — using forensics heuristics", {"disabled": True, "cloud": True}
    try:
        import torch
        _load_image()
        slot = _slots["image"]
        device = next(slot["model"].parameters()).device
        rgb = img.convert("RGB")
        inputs = {k: v.to(device) for k, v in slot["processor"](images=rgb, return_tensors="pt").items()}
        with torch.no_grad():
            logits = slot["model"](**inputs).logits
        probs = _softmax_probs(logits, slot["model"])
        ai_prob, label = _ai_from_probs(
            probs,
            ("artificial", "ai", "fake", "generated", "synthetic", "sdxl"),
            ("human", "real", "natural", "photo", "authentic"),
        )
        detail = f"Neural network ({IMAGE_MODEL}): {ai_prob:.0%} AI-generated"
        return ai_prob, detail, {"model": IMAGE_MODEL, "ai_probability": round(ai_prob, 4), "predicted_class": label}
    except Exception as exc:
        return 0.5, f"ML detector unavailable: {exc}", {"error": str(exc)}


def _text_chunks(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= TEXT_CHUNK:
        return [text] if text else []
    parts, buf = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if len(buf) + len(sent) + 1 <= TEXT_CHUNK:
            buf = f"{buf} {sent}".strip()
        else:
            if buf:
                parts.append(buf)
            buf = sent[:TEXT_CHUNK]
    if buf:
        parts.append(buf)
    return parts or [text[:TEXT_CHUNK]]


def predict_ai_text(text: str) -> tuple[float, str, dict[str, Any]]:
    from app.config import ML_ENABLED
    text = text.strip()
    if not ML_ENABLED:
        return 0.45, "Neural model disabled in cloud mode — using text heuristics", {"disabled": True, "cloud": True}
    if len(text) < 20:
        return 0.35, "Text too short for neural network analysis", {"skipped": True}
    try:
        import torch
        _load_text()
        slot = _slots["text"]
        device = next(slot["model"].parameters()).device
        scores = []
        for chunk in _text_chunks(text):
            inputs = slot["processor"](chunk, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = slot["model"](**inputs).logits
            probs = _softmax_probs(logits, slot["model"])
            fake, _ = _ai_from_probs(probs, ("fake", "ai", "gpt", "generated", "label_1"), ("real", "human", "label_0"))
            scores.append(fake)
        peak = max(scores)
        combined = 0.65 * peak + 0.35 * (sum(scores) / len(scores))
        detail = f"Neural network ({TEXT_MODEL}): {combined:.0%} AI ({len(scores)} segment(s))"
        return combined, detail, {
            "model": TEXT_MODEL,
            "ai_probability": round(combined, 4),
            "segments": len(scores),
            "peak_segment_score": round(peak, 4),
        }
    except Exception as exc:
        return 0.5, f"Text ML unavailable: {exc}", {"error": str(exc)}
