"""SQLite persistence + analytics."""

import json
import sqlite3
from datetime import datetime, timezone

from app.config import DATA_DIR
from app.models import AnalysisResult, ContentType, Signal, Verdict

DB_PATH = DATA_DIR / "findai.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                scan_id TEXT PRIMARY KEY,
                content_type TEXT NOT NULL,
                filename TEXT,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasons TEXT NOT NULL,
                signals TEXT NOT NULL,
                limitations TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_created ON scans(created_at DESC)")


def _row_to_result(row: sqlite3.Row) -> AnalysisResult:
    return AnalysisResult(
        scan_id=row["scan_id"],
        content_type=ContentType(row["content_type"]),
        filename=row["filename"],
        verdict=Verdict(row["verdict"]),
        confidence=row["confidence"],
        reasons=json.loads(row["reasons"]),
        signals=[Signal(**s) for s in json.loads(row["signals"])],
        limitations=json.loads(row["limitations"]),
        metadata=json.loads(row["metadata"]),
    )


def save_scan(result: AnalysisResult) -> None:
    row = {
        "scan_id": result.scan_id,
        "content_type": result.content_type.value,
        "filename": result.filename,
        "verdict": result.verdict.value,
        "confidence": result.confidence,
        "reasons": json.dumps(result.reasons),
        "signals": json.dumps([s.to_dict() for s in result.signals]),
        "limitations": json.dumps(result.limitations),
        "metadata": json.dumps(result.metadata, default=str),
        "created_at": result.metadata.get("scanned_at") or datetime.now(timezone.utc).isoformat(),
    }
    with _connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO scans
               (scan_id, content_type, filename, verdict, confidence, reasons, signals, limitations, metadata, created_at)
               VALUES (:scan_id, :content_type, :filename, :verdict, :confidence,
                       :reasons, :signals, :limitations, :metadata, :created_at)""",
            row,
        )


def get_scan(scan_id: str) -> AnalysisResult | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,)).fetchone()
    return _row_to_result(row) if row else None


def list_scans(limit: int = 50) -> list[AnalysisResult]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM scans ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_result(r) for r in rows]


def delete_scan(scan_id: str) -> bool:
    with _connect() as conn:
        return conn.execute("DELETE FROM scans WHERE scan_id = ?", (scan_id,)).rowcount > 0


def clear_history() -> int:
    with _connect() as conn:
        return conn.execute("DELETE FROM scans").rowcount


def scan_count() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM scans").fetchone()
    return int(row["c"]) if row else 0


def scan_stats() -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM scans").fetchone()["c"]
        by_verdict = {r["verdict"]: r["c"] for r in conn.execute(
            "SELECT verdict, COUNT(*) AS c FROM scans GROUP BY verdict"
        )}
        by_type = {r["content_type"]: r["c"] for r in conn.execute(
            "SELECT content_type, COUNT(*) AS c FROM scans GROUP BY content_type"
        )}
        avg_conf = conn.execute("SELECT AVG(confidence) AS a FROM scans").fetchone()["a"]
        recent = conn.execute(
            "SELECT scan_id, filename, verdict, confidence, content_type, created_at FROM scans ORDER BY created_at DESC LIMIT 8"
        ).fetchall()
    return {
        "total": total,
        "by_verdict": by_verdict,
        "by_type": by_type,
        "avg_confidence": round(float(avg_conf or 0) * 100),
        "recent": [dict(r) for r in recent],
    }
