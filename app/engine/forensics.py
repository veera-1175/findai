"""Forensic helpers: image edits, text entropy, audio splice/voice."""

from __future__ import annotations

import io
import math
import re
from collections import Counter

import numpy as np
from PIL import Image

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# --- Image ELA / edits ---

def _ela_map(img: Image.Image, quality: int = 90) -> np.ndarray:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recon = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
    orig = np.array(img.convert("RGB"), dtype=np.float32)
    return np.mean(np.abs(orig - recon), axis=2)


def global_ela(img: Image.Image) -> tuple[float, str]:
    try:
        std_diff = float(np.std(_ela_map(img)))
        if std_diff > 8:
            return 0.62, f"Global compression inconsistency (σ={std_diff:.1f})"
        return 0.18, "Global compression patterns appear normal"
    except Exception:
        return 0.25, "Could not perform compression analysis"


def block_edit_analysis(img: Image.Image, grid: int = 4) -> tuple[float, str]:
    try:
        ela = _ela_map(img.convert("RGB").resize((512, 512)))
        h, w = ela.shape
        bh, bw = h // grid, w // grid
        blocks = [float(np.mean(ela[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw])) for r in range(grid) for c in range(grid)]
        mean_all, std_b = float(np.mean(blocks)), float(np.std(blocks))
        spread = max(blocks) - min(blocks)
        hot = sum(1 for b in blocks if b > mean_all + std_b * 1.5)
        if hot >= 2 and spread > 6:
            return min(0.92, 0.55 + hot * 0.08), f"Localized edit — {hot} region(s) (spread={spread:.1f})"
        if std_b > 4.5 and spread > 4:
            return 0.65, f"Uneven compression across blocks (σ={std_b:.1f})"
        return 0.15, "No localized compression anomalies"
    except Exception:
        return 0.3, "Block edit analysis failed"


def noise_inconsistency(img: Image.Image, grid: int = 4) -> tuple[float, str]:
    try:
        from scipy.ndimage import uniform_filter
        gray = np.array(img.convert("L").resize((512, 512)), dtype=np.float32)
        residual = gray - uniform_filter(gray, size=3)
        h, w = residual.shape
        bh, bw = h // grid, w // grid
        stds = [float(np.std(residual[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw])) for r in range(grid) for c in range(grid)]
        cv = float(np.std(stds) / (np.mean(stds) + 1e-6))
        if cv > 0.45 and (max(stds) - min(stds)) > 2.5:
            return min(0.88, 0.5 + cv * 0.4), f"Inconsistent noise (CV={cv:.2f}) — splicing/inpaint"
        if cv > 0.32:
            return 0.55, f"Mild noise inconsistency (CV={cv:.2f})"
        return 0.12, f"Noise consistent (CV={cv:.2f})"
    except Exception:
        return 0.25, "Noise analysis failed"


def analyze_edits(img: Image.Image) -> list[tuple[str, str, float, str]]:
    checks = [
        ("block_ela", "Localized compression (ELA)", block_edit_analysis(img)),
        ("noise_inconsistency", "Noise consistency", noise_inconsistency(img)),
    ]
    return [(n, lbl, s, d) for n, lbl, (s, d) in checks]


# --- Text entropy ---

def ai_likelihood_from_entropy(text: str) -> tuple[float, str]:
    if len(text) < 80:
        return 0.3, "Text too short for entropy analysis"
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    if len(words) < 5:
        return 0.3, "Not enough words"
    counts = Counter(words)
    total = len(words)
    w_ent = -sum((c / total) * math.log2(c / total) for c in counts.values())
    grams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    rep = sum(1 for c in Counter(grams).values() if c > 1) / max(len(set(grams)), 1)
    score, details = 0.0, []
    if 4.0 <= w_ent <= 5.8:
        score += 0.25
        details.append(f"word entropy {w_ent:.2f} in AI range")
    if rep > 0.15:
        score += 0.35
        details.append(f"repeated phrases ({rep:.0%})")
    return min(1.0, score), "; ".join(details) if details else f"entropy natural ({w_ent:.2f})"


# --- Audio forensics ---

def _segment_rms(samples: np.ndarray, sr: int, seg_sec: float = 0.4) -> list[float]:
    seg_len = max(int(sr * seg_sec), 1)
    return [float(np.sqrt(np.mean(samples[i:i + seg_len] ** 2))) for i in range(0, len(samples) - seg_len, seg_len)]


def detect_splice_edits(samples: np.ndarray, sr: int) -> tuple[float, str]:
    rms = _segment_rms(samples, sr)
    if len(rms) < 4:
        return 0.25, "Audio too short for splice analysis"
    jumps = [abs(rms[i] - rms[i - 1]) for i in range(1, len(rms))]
    ratio = max(jumps) / (float(np.mean(jumps)) + 1e-9)
    if ratio > 4 and max(jumps) > 0.05:
        return 0.72, f"Splice boundary detected (ratio {ratio:.1f})"
    if ratio > 2.5:
        return 0.55, f"Mild segment inconsistencies (ratio {ratio:.1f})"
    return 0.12, "No splice boundaries detected"


def detect_synthetic_voice(samples: np.ndarray, sr: int) -> tuple[float, str]:
    scores, details = [], []
    if HAS_LIBROSA:
        try:
            pitches, mags = librosa.piptrack(y=samples, sr=sr, threshold=0.1)
            pvals = [float(pitches[mags[:, t].argmax(), t]) for t in range(pitches.shape[1])
                     if pitches[mags[:, t].argmax(), t] > 50]
            if len(pvals) > 10:
                cv = float(np.std(pvals) / (np.mean(pvals) + 1e-9))
                if cv < 0.08:
                    scores.append(0.68)
                    details.append(f"stable pitch CV={cv:.3f}")
        except Exception:
            pass
    fft = np.abs(np.fft.rfft(samples[: min(len(samples), sr * 3)]))
    fft = fft[fft > 0]
    if len(fft) > 10:
        flat = float(np.exp(np.mean(np.log(fft + 1e-10))) / (np.mean(fft) + 1e-10))
        if flat > 0.12:
            scores.append(0.62)
            details.append(f"spectral flatness {flat:.3f}")
    if not scores:
        return 0.2, "Voice analysis inconclusive"
    combined = max(scores) * 0.7 + (sum(scores) / len(scores)) * 0.3
    return min(0.92, combined), "Synthetic voice: " + "; ".join(details)


def analyze_audio_forensics(samples: np.ndarray, sr: int) -> list[tuple[str, str, float, str]]:
    s, d = detect_synthetic_voice(samples, sr)
    sp, sd = detect_splice_edits(samples, sr)
    return [("ml_audio", "Voice analysis", s, d), ("audio_splice", "Splice detection", sp, sd)]
