"""Basic tests — run: python -m pytest tests/"""

from app.engine.analyze import analyze_text
from app.engine.fusion import fuse_signals
from app.models import Signal, Verdict

AI_SAMPLE = (
    "Moreover, it is worth noting that in today's digital age we must delve into "
    "comprehensive seamless solutions. Furthermore this robust integration leverages "
    "multifaceted technology across the ever-evolving landscape."
)

HUMAN_SAMPLE = (
    "I missed the bus today so I walked home. My shoes got wet because it started raining "
    "halfway. Called my friend and we just talked about nothing for an hour. Pretty normal Tuesday."
)


def test_ai_text_detected():
    result = analyze_text("t1", AI_SAMPLE)
    assert result.verdict in (Verdict.AI_GENERATED, Verdict.UNCERTAIN, Verdict.AI_EDITED)
    assert result.confidence > 0.4
    assert len(result.signals) >= 2


def test_human_text_lower_risk():
    result = analyze_text("t2", HUMAN_SAMPLE)
    assert result.verdict in (Verdict.AUTHENTIC, Verdict.UNCERTAIN)
    assert max(s.score for s in result.signals) < 0.8


def test_fusion_empty_signals():
    verdict, conf, reasons = fuse_signals([])
    assert verdict == Verdict.UNCERTAIN
    assert conf == 0.4


def test_fusion_authentic():
    signals = [Signal("a", "A", 0.1, "ok"), Signal("b", "B", 0.15, "ok")]
    verdict, conf, _ = fuse_signals(signals, "image")
    assert verdict == Verdict.AUTHENTIC


def test_fusion_image_ml_ai():
    signals = [
        Signal("ml_detector", "ML", 0.91, "91% AI"),
        Signal("block_ela", "Edit", 0.2, "no edit"),
    ]
    verdict, conf, _ = fuse_signals(signals, "image")
    assert verdict == Verdict.AI_GENERATED
    assert conf > 0.7


def test_fusion_text_ml():
    signals = [
        Signal("ml_text_detector", "ML", 0.88, "88% AI text"),
        Signal("burstiness", "Burst", 0.3, "ok"),
    ]
    verdict, conf, _ = fuse_signals(signals, "text")
    assert verdict == Verdict.AI_GENERATED
    assert conf > 0.7


def test_fusion_audio_synthetic():
    signals = [
        Signal("ml_audio", "Voice", 0.78, "synthetic"),
        Signal("audio_splice", "Splice", 0.2, "clean"),
    ]
    verdict, conf, _ = fuse_signals(signals, "audio")
    assert verdict == Verdict.AI_GENERATED


def test_fusion_audio_edited():
    signals = [
        Signal("ml_audio", "Voice", 0.3, "real voice"),
        Signal("audio_splice", "Splice", 0.8, "spliced"),
    ]
    verdict, _, _ = fuse_signals(signals, "audio")
    assert verdict == Verdict.AI_EDITED


def test_fusion_image_edited():
    signals = [
        Signal("ml_detector", "ML", 0.25, "real base"),
        Signal("block_ela", "Edit", 0.78, "localized edit"),
    ]
    verdict, conf, _ = fuse_signals(signals, "image")
    assert verdict == Verdict.AI_EDITED
    assert conf > 0.5
