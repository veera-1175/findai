"""Content type routing."""

from app.config import ALLOWED_EXTENSIONS
from app.engine.analyze import analyze_audio, analyze_document, analyze_image, analyze_text, analyze_video
from app.models import AnalysisResult, ContentType

_MAP = (
    (ContentType.IMAGE, ALLOWED_EXTENSIONS["image"]),
    (ContentType.VIDEO, ALLOWED_EXTENSIONS["video"]),
    (ContentType.AUDIO, ALLOWED_EXTENSIONS["audio"]),
    (ContentType.DOCUMENT, ALLOWED_EXTENSIONS["document"]),
)


def detect_content_type(filename: str | None, text_input: str | None) -> ContentType:
    if text_input and text_input.strip():
        return ContentType.TEXT
    if not filename:
        raise ValueError("No file or text provided")
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    for ctype, exts in _MAP:
        if ext in exts:
            return ctype
    raise ValueError(f"Unsupported file type: {ext or 'unknown'}")


def run_analysis(
    scan_id: str,
    content_type: ContentType,
    file_path: str | None = None,
    filename: str | None = None,
    text_input: str | None = None,
) -> AnalysisResult:
    routes = {
        ContentType.TEXT: lambda: analyze_text(scan_id, text_input or "", filename),
        ContentType.IMAGE: lambda: analyze_image(scan_id, file_path or "", filename or ""),
        ContentType.VIDEO: lambda: analyze_video(scan_id, file_path or "", filename or ""),
        ContentType.AUDIO: lambda: analyze_audio(scan_id, file_path or "", filename or ""),
        ContentType.DOCUMENT: lambda: analyze_document(scan_id, file_path or "", filename or ""),
    }
    fn = routes.get(content_type)
    if not fn:
        raise ValueError(f"Unknown content type: {content_type}")
    return fn()
