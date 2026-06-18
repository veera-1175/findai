import json
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import __version__
from app.config import BASE_DIR, DEBUG, IS_CLOUD, PRELOAD_ML
from app.core.scan import perform_scan
from app.db import clear_history, delete_scan, get_scan, init_db, list_scans, scan_count, scan_stats
from app.demo import DEMO_SAMPLES
from app.engine.ml import all_model_status, preload_all


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    if PRELOAD_ML:
        threading.Thread(target=preload_all, daemon=True, name="ml-preload").start()
    yield


app = FastAPI(
    title="FindAI",
    description="Multimodal AI & fake content detector — free, cloud-ready, open-source",
    version=__version__,
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
templates.env.globals["version"] = __version__
templates.env.filters["tojson"] = lambda v, indent=0: json.dumps(v, indent=indent, default=str)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")


def _render(request: Request, name: str, context: dict | None = None, status_code: int = 200):
    return templates.TemplateResponse(request, name, context or {}, status_code=status_code)


@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": exc.detail}, status_code=404)
    return _render(request, "error.html", {"code": 404, "message": exc.detail or "Page not found"}, 404)


@app.exception_handler(500)
async def server_error(request: Request, exc: Exception):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": str(exc) if DEBUG else "Internal server error"}, status_code=500)
    return _render(request, "error.html", {"code": 500, "message": "Something went wrong. Please try again."}, 500)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return _render(request, "index.html", {
        "recent_scans": list_scans(6),
        "stats": scan_stats(),
        "demo_samples": DEMO_SAMPLES,
        "models": all_model_status(),
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return _render(request, "dashboard.html", {
        "stats": scan_stats(),
        "models": all_model_status(),
        "total_scans": scan_count(),
    })


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return _render(request, "about.html", {"models": all_model_status(), "is_cloud": IS_CLOUD})


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    return _render(request, "history.html", {"scans": list_scans(100), "total_scans": scan_count()})


@app.post("/history/clear")
async def history_clear():
    clear_history()
    return RedirectResponse("/", status_code=303)


@app.post("/history/{scan_id}/delete")
async def history_delete(scan_id: str):
    delete_scan(scan_id)
    return RedirectResponse("/history", status_code=303)


@app.get("/result/{scan_id}", response_class=HTMLResponse)
async def result_page(request: Request, scan_id: str):
    result = get_scan(scan_id)
    if not result:
        raise HTTPException(404, "Scan not found — it may have been cleared from history.")
    return _render(request, "result.html", {"result": result})


@app.get("/result/{scan_id}/export")
async def export_result(scan_id: str):
    result = get_scan(scan_id)
    if not result:
        raise HTTPException(404, "Scan not found")
    return JSONResponse(result.to_dict())


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_web(request: Request, file: UploadFile | None = File(None), text: str = Form("")):
    return _render(request, "result.html", {"result": perform_scan(file=file, text=text)})


@app.post("/api/analyze")
async def analyze_api(file: UploadFile | None = File(None), text: str = Form("")):
    return JSONResponse(perform_scan(file=file, text=text).to_dict())


@app.get("/api/scans")
async def api_list_scans(limit: int = 20):
    return [r.to_dict() for r in list_scans(min(limit, 100))]


@app.get("/api/scans/{scan_id}")
async def api_get_scan(scan_id: str):
    result = get_scan(scan_id)
    if not result:
        raise HTTPException(404, "Scan not found")
    return result.to_dict()


@app.get("/api/stats")
async def api_stats():
    return scan_stats()


@app.post("/demo/{demo_id}", response_class=HTMLResponse)
async def run_demo(request: Request, demo_id: str):
    sample = next((s for s in DEMO_SAMPLES if s["id"] == demo_id), None)
    if not sample:
        raise HTTPException(404, "Demo sample not found")
    return _render(request, "result.html", {"result": perform_scan(text=sample["text"])})


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": __version__,
        "engine": "findai",
        "cloud": IS_CLOUD,
        "paid_apis": False,
        "total_scans": scan_count(),
        "models": all_model_status(),
    }
