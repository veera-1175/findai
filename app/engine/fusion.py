from app.models import Verdict, Signal

ML_IMAGE = {"ml_detector", "ml_frame_avg"}
ML_TEXT = {"ml_text_detector", "text_ml_text_detector"}
HEUR_AI_TEXT = {"ai_phrases", "ai_vocabulary", "uniform_paragraphs", "text_ai_phrases", "text_ai_vocabulary"}
HUMAN_TEXT = {"human_markers", "text_human_markers"}
EDIT_IMAGE = {"ela", "block_ela", "noise_inconsistency", "edge_inconsistency", "frame_edit"}
AUDIO_SYNTH = {"ml_audio", "spectral", "amplitude", "pitch", "synthetic_voice", "zcr", "rolloff", "noise"}
AUDIO_SPLICE = {"audio_splice", "segment_jump"}
# Short-text heuristics that misfire on casual writing — don't use alone for AI verdict
SHORT_TEXT_HEURISTICS = {"burstiness", "entropy", "word_uniformity", "text_burstiness", "text_entropy"}


def _max_named(signals: list[Signal], names: set[str]) -> float | None:
    vals = [s.score for s in signals if s.name in names]
    return max(vals) if vals else None


def _avg_named(signals: list[Signal], names: set[str]) -> float | None:
    vals = [s.score for s in signals if s.name in names]
    return sum(vals) / len(vals) if vals else None


def fuse_signals(signals: list[Signal], content_type: str = "unknown") -> tuple[Verdict, float, list[str]]:
    if not signals:
        return Verdict.UNCERTAIN, 0.4, ["Not enough data to analyze this content."]

    reasons: list[str] = []
    for s in sorted(signals, key=lambda x: x.score, reverse=True)[:6]:
        if s.score >= 0.32:
            reasons.append(s.detail)
    if not reasons:
        reasons.append("No strong manipulation indicators detected.")

    ml_img = _max_named(signals, ML_IMAGE)
    ml_txt = _max_named(signals, ML_TEXT)
    edit_max = _max_named(signals, EDIT_IMAGE) or 0.0
    audio_synth = _max_named(signals, AUDIO_SYNTH)
    audio_splice = _max_named(signals, AUDIO_SPLICE)
    heur_ai = _max_named(signals, HEUR_AI_TEXT) or 0.0
    human_score = _min_human(signals)

    scores = [s.score for s in signals]
    combined = 0.4 * (sum(scores) / len(scores)) + 0.6 * max(scores)

    if content_type == "image" and ml_img is not None:
        return _fuse_image_video(ml_img, edit_max, reasons, "image")

    if content_type == "video" and ml_img is not None:
        temporal = _max_named(signals, {"temporal"}) or 0.0
        return _fuse_image_video(ml_img, max(edit_max, temporal * 0.85), reasons, "video")

    if content_type in ("text", "document"):
        return _fuse_text(ml_txt, heur_ai, human_score, combined, reasons)

    if content_type == "audio":
        return _fuse_audio(audio_synth, audio_splice, combined, reasons)

    if ml_img is not None:
        return _fuse_image_video(ml_img, edit_max, reasons, content_type)

    return _fuse_heuristic(combined, signals, content_type, reasons)


def _min_human(signals: list[Signal]) -> float | None:
    vals = [s.score for s in signals if s.name in HUMAN_TEXT]
    return min(vals) if vals else None


def _fuse_text(
    ml_txt: float | None,
    heur_ai: float,
    human_score: float | None,
    combined: float,
    reasons: list[str],
) -> tuple[Verdict, float, list[str]]:
    """ML model leads; heuristics only support when they agree."""

  # Strong human signals + low ML → authentic
    if ml_txt is not None and ml_txt <= 0.28:
        if heur_ai < 0.45:
            conf = min(0.94, 0.62 + (1.0 - ml_txt) * 0.3)
            reasons.insert(0, f"Neural model: {1 - ml_txt:.0%} likely human-written")
            return Verdict.AUTHENTIC, conf, reasons[:6]

    # Strong AI from model OR model + phrase patterns agree
    if ml_txt is not None and (ml_txt >= 0.68 or (ml_txt >= 0.50 and heur_ai >= 0.50)):
        conf = min(0.96, 0.55 + max(ml_txt, heur_ai) * 0.4)
        reasons.insert(0, f"Neural model: {ml_txt:.0%} likely AI-generated")
        return Verdict.AI_GENERATED, conf, reasons[:6]

    # Heuristics alone (very strong AI phrases) when ML unavailable
    if ml_txt is None and heur_ai >= 0.65:
        return Verdict.AI_GENERATED, min(0.9, 0.5 + heur_ai * 0.4), reasons[:6]

    # ML moderate-high with supporting heuristics
    if ml_txt is not None and ml_txt >= 0.45 and heur_ai >= 0.35:
        conf = min(0.88, 0.45 + (ml_txt + heur_ai) * 0.25)
        reasons.insert(0, f"Model + writing patterns suggest AI ({ml_txt:.0%})")
        return Verdict.AI_GENERATED, conf, reasons[:6]

    # ML low + human markers
    if ml_txt is not None and ml_txt <= 0.40 and human_score is not None and human_score <= 0.25:
        conf = min(0.92, 0.6 + (1.0 - ml_txt) * 0.28)
        reasons.insert(0, "Personal, natural writing style detected")
        return Verdict.AUTHENTIC, conf, reasons[:6]

    if ml_txt is not None and ml_txt <= 0.40 and heur_ai < 0.40:
        return Verdict.AUTHENTIC, min(0.88, 0.55 + (1.0 - ml_txt) * 0.3), reasons[:6]

    if combined < 0.38 and heur_ai < 0.40:
        return Verdict.AUTHENTIC, min(0.85, 1.0 - combined), reasons[:6]

    return Verdict.UNCERTAIN, 0.52, reasons[:6]


def _fuse_audio(
    audio_synth: float | None,
    audio_splice: float | None,
    combined: float,
    reasons: list[str],
) -> tuple[Verdict, float, list[str]]:
    synth = audio_synth if audio_synth is not None else combined
    splice = audio_splice if audio_splice is not None else 0.0

    if synth >= 0.58:
        conf = min(0.94, 0.52 + synth * 0.42)
        reasons.insert(0, f"Voice analysis: {synth:.0%} likely synthetic")
        return Verdict.AI_GENERATED, conf, reasons[:6]
    if splice >= 0.55 and synth < 0.42:
        conf = min(0.92, 0.5 + splice * 0.42)
        reasons.insert(0, "Edit boundaries detected in audio")
        return Verdict.AI_EDITED, conf, reasons[:6]
    if synth <= 0.32 and splice < 0.35:
        return Verdict.AUTHENTIC, min(0.92, 0.58 + (1.0 - synth) * 0.32), reasons[:6]
    if splice >= 0.48:
        return Verdict.AI_EDITED, min(0.86, 0.45 + splice * 0.38), reasons[:6]
    return Verdict.UNCERTAIN, 0.54, reasons[:6]


def _fuse_image_video(
    ml_ai: float, edit_max: float, reasons: list[str], kind: str
) -> tuple[Verdict, float, list[str]]:
    if ml_ai >= 0.58:
        conf = min(0.96, 0.54 + ml_ai * 0.42)
        reasons.insert(0, f"Image model: {ml_ai:.0%} likely AI-generated")
        return Verdict.AI_GENERATED, conf, reasons[:6]
    if edit_max >= 0.55 and ml_ai < 0.42:
        conf = min(0.93, 0.52 + edit_max * 0.4)
        reasons.insert(0, "Localized edit/manipulation detected")
        return Verdict.AI_EDITED, conf, reasons[:6]
    if ml_ai <= 0.35 and edit_max < 0.40:
        conf = min(0.94, 0.6 + (1.0 - ml_ai) * 0.34)
        reasons.insert(0, f"Image model: {1 - ml_ai:.0%} likely authentic")
        return Verdict.AUTHENTIC, conf, reasons[:6]
    if edit_max >= 0.48 and ml_ai < 0.50:
        return Verdict.AI_EDITED, min(0.88, 0.48 + edit_max * 0.38), reasons[:6]
    if ml_ai >= 0.45:
        return Verdict.UNCERTAIN, 0.55, reasons[:6]
    return Verdict.AUTHENTIC, min(0.82, 0.55 + (1.0 - ml_ai) * 0.25), reasons[:6]


def _fuse_heuristic(
    combined: float, signals: list[Signal], content_type: str, reasons: list[str]
) -> tuple[Verdict, float, list[str]]:
    high = [s for s in signals if s.score >= 0.58 and s.name not in SHORT_TEXT_HEURISTICS]
    edit_max = _max_named(signals, EDIT_IMAGE | AUDIO_SPLICE) or 0.0

    if combined < 0.36:
        return Verdict.AUTHENTIC, min(0.9, 1.0 - combined), reasons[:6]
    if len(high) >= 2 or (combined >= 0.65 and high):
        if edit_max >= 0.5:
            return Verdict.AI_EDITED, min(0.9, combined + 0.1), reasons[:6]
        return Verdict.AI_GENERATED, min(0.9, combined + 0.08), reasons[:6]
    if combined >= 0.55 and high:
        return Verdict.AI_GENERATED, combined, reasons[:6]
    return Verdict.UNCERTAIN, 0.52, reasons[:6]
