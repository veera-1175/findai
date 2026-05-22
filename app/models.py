from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    AUTHENTIC = "authentic"
    UNCERTAIN = "uncertain"
    AI_GENERATED = "ai_generated"
    AI_EDITED = "ai_edited"


class ContentType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    DOCUMENT = "document"


@dataclass
class Signal:
    name: str
    label: str
    score: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "score": round(self.score, 3),
            "detail": self.detail,
        }


@dataclass
class AnalysisResult:
    scan_id: str
    content_type: ContentType
    filename: str | None
    verdict: Verdict
    confidence: float
    reasons: list[str]
    signals: list[Signal]
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "content_type": self.content_type.value,
            "filename": self.filename,
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 3),
            "reasons": self.reasons,
            "signals": [s.to_dict() for s in self.signals],
            "limitations": self.limitations,
            "metadata": self.metadata,
        }

    @property
    def verdict_label(self) -> str:
        labels = {
            Verdict.AUTHENTIC: "Likely Authentic",
            Verdict.UNCERTAIN: "Uncertain",
            Verdict.AI_GENERATED: "Likely AI-Generated",
            Verdict.AI_EDITED: "Likely AI-Edited / Fake",
        }
        return labels[self.verdict]

    @property
    def verdict_color(self) -> str:
        colors = {
            Verdict.AUTHENTIC: "emerald",
            Verdict.UNCERTAIN: "amber",
            Verdict.AI_GENERATED: "rose",
            Verdict.AI_EDITED: "orange",
        }
        return colors[self.verdict]

    @property
    def confidence_percent(self) -> int:
        return int(round(self.confidence * 100))

    @property
    def preview(self) -> dict:
        return self.metadata.get("preview") or {}

    @property
    def scanned_at(self) -> str:
        raw = self.metadata.get("scanned_at", "")
        return raw[:19].replace("T", " ") if raw else ""

    @property
    def technical_info(self) -> list[dict[str, str]]:
        """Structured rows for the analysis details panel."""
        rows: list[dict[str, str]] = []
        p = self.preview

        rows.append({"label": "Content", "value": p.get("label") or self.filename or "Text"})
        rows.append({"label": "Type", "value": self.content_type.value.replace("_", " ").title()})

        if p.get("file_size_label"):
            rows.append({"label": "File size", "value": p["file_size_label"]})
        if p.get("dimensions"):
            rows.append({"label": "Dimensions", "value": str(p["dimensions"])})
        if p.get("duration"):
            rows.append({"label": "Duration", "value": f"{p['duration']} sec"})
        if p.get("resolution"):
            rows.append({"label": "Resolution", "value": str(p["resolution"])})
        if p.get("char_count"):
            rows.append({"label": "Characters", "value": str(p["char_count"])})
        if p.get("word_count"):
            rows.append({"label": "Words", "value": str(p["word_count"])})
        if self.scanned_at:
            rows.append({"label": "Checked at", "value": self.scanned_at})
        rows.append({"label": "Scan ID", "value": self.scan_id})

        ml = self.metadata.get("ml") or {}
        if ml.get("model"):
            rows.append({"label": "Model used", "value": ml["model"]})

        return rows
