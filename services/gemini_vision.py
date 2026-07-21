"""Rate-limited Gemini Vision analysis with resumable SQLite persistence."""
import base64
import json
import logging
import os
import random
import sqlite3
import time
from typing import Callable, Optional
from urllib import error, parse, request

from core import database

LOGGER = logging.getLogger(__name__)
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_INTERVAL_SECONDS = 4.5
MAX_RETRIES = 5
FIELDS = ("scene", "characters", "dialogue", "narration", "actions", "emotion", "camera_suggestion")


def _api_request(url: str, key: str, payload: dict | None = None) -> dict:
    """Perform one Gemini request, retrying only temporary service failures."""
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url, data=body, headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    for attempt in range(MAX_RETRIES):
        try:
            with request.urlopen(req, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError(f"Gemini HTTP {exc.code}: {response_body[:400]}") from exc
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Gemini rate/service limit persisted after {MAX_RETRIES} attempts (HTTP {exc.code}).") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else min(60.0, 2 ** attempt)
            delay += random.uniform(0.0, 0.75)
            LOGGER.warning("Gemini HTTP %s; retrying in %.1fs (attempt %s/%s).", exc.code, delay, attempt + 1, MAX_RETRIES)
            time.sleep(delay)
        except error.URLError as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Gemini network failure: {exc.reason}") from exc
            delay = min(30.0, 2 ** attempt) + random.uniform(0.0, 0.75)
            LOGGER.warning("Gemini network failure; retrying in %.1fs.", delay)
            time.sleep(delay)
    raise RuntimeError("Gemini request ended unexpectedly.")


def resolve_model(api_key: str | None = None) -> str:
    """Resolve an installed generateContent model before starting a large job.

    This preflight converts a previous per-panel 404 storm into one actionable
    error. A configured model must be returned by ``models.list`` for the key.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY before analyzing panels.")
    result = _api_request(f"{API_ROOT}/models", key)
    available = {model.get("name", ""): model for model in result.get("models", []) if "generateContent" in model.get("supportedGenerationMethods", [])}
    configured = os.getenv("GEMINI_MODEL", "").strip()
    preferred = [configured] if configured else ["models/gemini-2.5-flash", "models/gemini-2.0-flash", "models/gemini-3.5-flash"]
    for model in preferred:
        normalized = model if model.startswith("models/") else f"models/{model}"
        if normalized in available:
            return normalized
    names = ", ".join(sorted(available)[:12]) or "none"
    raise RuntimeError(f"No configured Gemini vision model is available for this API key. Set GEMINI_MODEL to one of: {names}")


def _prompt() -> str:
    return """Analyze this comic panel. Return only JSON with: scene (string), characters (array of strings), dialogue (array of {speaker,text}), narration (string), actions (array), emotion (string), camera_suggestion (static, pan_left, pan_right, zoom_in, or zoom_out). Transcribe only clearly visible dialogue and use empty values when unavailable."""


def analyze_panel(image_path: str, model: str, api_key: str | None = None) -> dict:
    """Analyze a single PNG panel using a model verified by :func:`resolve_model`."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY before analyzing panels.")
    with open(image_path, "rb") as source:
        encoded = base64.b64encode(source.read()).decode("ascii")
    payload = {"contents": [{"parts": [{"text": _prompt()}, {"inlineData": {"mimeType": "image/png", "data": encoded}}]}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}
    response = _api_request(f"{API_ROOT}/{parse.quote(model, safe='/')}:generateContent", key, payload)
    try:
        analysis = json.loads(response["candidates"][0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Gemini returned no valid JSON analysis for this panel.") from exc
    if not isinstance(analysis, dict):
        raise ValueError("Gemini returned a non-object analysis.")
    for field in FIELDS:
        analysis.setdefault(field, [] if field in ("characters", "dialogue", "actions") else "")
    if not isinstance(analysis["dialogue"], list):
        raise ValueError("Gemini returned invalid dialogue data.")
    return analysis


def analyze_all_panels(project_name: str, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> dict:
    """Resume unfinished panel analysis sequentially, with throttling and commits.

    Successful rows are committed immediately. Failed rows remain unanalysed so
    a later run resumes them, while the rest of the project continues.
    """
    model = resolve_model()
    interval = max(0.0, float(os.getenv("GEMINI_MIN_REQUEST_INTERVAL", DEFAULT_INTERVAL_SECONDS)))
    panels = [panel for panel in database.get_all_panels(project_name) if not panel.get("analysis_json")]
    if not panels:
        database.update_project_state(project_name, "ANALYZED")
        return {"total": 0, "analyzed": 0, "skipped": 0, "errors": [], "model": model}
    conn = sqlite3.connect(database.get_db_path(project_name)); completed = 0; errors: list[str] = []
    try:
        for index, panel in enumerate(panels, 1):
            started = time.monotonic()
            try:
                analysis = analyze_panel(panel["image_path"], model)
                conn.execute("UPDATE panels SET analysis_json=?, state='ANALYZED' WHERE id=?", (json.dumps(analysis, ensure_ascii=False), panel["id"]))
                conn.commit(); completed += 1
            except Exception as exc:
                LOGGER.warning("Panel %s analysis failed: %s", panel["id"], exc)
                errors.append(f"Panel {panel['id']}: {exc}")
            if progress_callback: progress_callback(index, len(panels), "analyzed" if not errors else "partial")
            remaining = interval - (time.monotonic() - started)
            if remaining > 0 and index < len(panels): time.sleep(remaining)
    finally:
        conn.close()
    if not errors:
        database.update_project_state(project_name, "ANALYZED")
    return {"total": len(panels), "analyzed": completed, "skipped": 0, "errors": errors, "model": model}
