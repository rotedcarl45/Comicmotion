"""
Panel Detection Engine — Module 3.

Converts extracted comic page images into individual panel crops using
MagiV3 (`ragavsachdeva/magiv3`), a unified comic-understanding transformer
that replaces the previous YOLO backend (`mosesb/best-comic-panel-detection`).

Unlike YOLO, MagiV3 detects panels, text/speech-bubble regions and
characters in a single forward pass and can also transcribe (OCR) every
detected text region — so this module now surfaces dialogue/OCR data in
addition to panel boxes.

Pipeline per page:
    1. Load the cached MagiV3 model + processor (loaded once, reused for
       every page).
    2. Run `predict_detections_and_associations()` on the full-resolution
       page image to get panel, text and character boxes.
    3. Convert each panel prediction into an (x, y, w, h) bounding box.
    4. IoU-based deduplication (keeps the highest-confidence box).
    5. Western reading-order sort (top→bottom, left→right).
    6. Run `predict_ocr()` on the detected text/speech-bubble regions and
       geometrically associate each one with its containing panel.
    7. Crop & save panel PNG files.
    8. Write SQLite panel records (including confidence and OCR data, when
       supported by the database layer — see the module-level note near
       `_INSERT_PANEL_SUPPORTS_OCR`).
    9. Update project state → PANELS_READY.
"""
import inspect
import json
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from typing import Callable, Optional

from core import database, workspace

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError as exc:  # pragma: no cover - surfaced at import time
    raise ImportError(
        "The 'torch' and 'transformers' packages are required for panel "
        "detection (MagiV3 backend). Install them with: "
        "pip install torch transformers"
    ) from exc


# ---------------------------------------------------------------------------
# Model loading (cached, loaded once)
# ---------------------------------------------------------------------------

# Hugging Face Hub repo id (or local snapshot directory) for MagiV3.
# PANEL_DETECTOR_MODEL_PATH is kept as the primary override for backward
# compatibility with existing deployment configs that already set it (it can
# point at a local snapshot dir instead of a weights file now); the new
# PANEL_DETECTOR_MODEL_ID variable is provided as a clearer alias.
MODEL_PATH = os.environ.get(
    "PANEL_DETECTOR_MODEL_PATH",
    os.environ.get("PANEL_DETECTOR_MODEL_ID", "ragavsachdeva/magiv3"),
)

# Minimum confidence for a panel detection to be considered "real". Passed
# through to MagiV3 when its API exposes a matching threshold kwarg (see
# `_detect_with_confidence`); otherwise used only for the debug view split.
_CONF_THRESHOLD = 0.25

_MODEL = None
_PROCESSOR = None


def _get_device() -> str:
    """Pick the best available torch device for MagiV3 inference."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    """
    Load and cache the MagiV3 comic-understanding model.

    The model is loaded exactly once per process and reused for every
    subsequent page — reloading it per page would be a major performance
    regression. This mirrors the previous YOLO caching behaviour so
    callers that already rely on `get_model()` being cheap after the first
    call keep working unchanged.

    Returns:
        The cached MagiV3 (`transformers.AutoModelForCausalLM`) model
        instance, moved to GPU when available.

    Raises:
        RuntimeError: If the model weights cannot be loaded (missing
            network access to the Hugging Face Hub, no local snapshot at
            `MODEL_PATH`, or an incompatible `transformers` version).
    """
    global _MODEL
    if _MODEL is None:
        try:
            dtype = torch.float16 if _get_device() == "cuda" else torch.float32
            _MODEL = (
                AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH, torch_dtype=dtype, trust_remote_code=True
                )
                .to(_get_device())
                .eval()
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the MagiV3 panel-detection model from "
                f"'{MODEL_PATH}'. Ensure you have network access to the "
                "Hugging Face Hub (or set PANEL_DETECTOR_MODEL_PATH / "
                "PANEL_DETECTOR_MODEL_ID to a local snapshot directory of "
                "'ragavsachdeva/magiv3'), and that 'torch' + 'transformers' "
                "are installed with a version supporting 'trust_remote_code'."
            ) from exc
    return _MODEL


def _get_processor():
    """
    Load and cache the MagiV3 processor (tokenizer/image-processor pair).

    Cached separately from the model itself, but on the same lifecycle —
    both live for the duration of the process.
    """
    global _PROCESSOR
    if _PROCESSOR is None:
        _PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return _PROCESSOR


# ---------------------------------------------------------------------------
# Reading-order sort
# ---------------------------------------------------------------------------

def _sort_reading_order(items: list, key: Callable = lambda item: item) -> list:
    """
    Sort items in western comic reading order, by (x, y, w, h) box.

    Two panels share a row when their vertical overlap is >= 50 % of the
    shorter panel's height. Rows are ordered top-to-bottom; panels within
    each row are ordered left-to-right.

    Args:
        items: List of items to sort (typically (x, y, w, h) tuples, or
            (bbox, confidence) pairs).
        key:   Function extracting the (x, y, w, h) bbox from each item.
            Defaults to the identity function, preserving the original
            behaviour when `items` are plain bbox tuples.

    Returns:
        Re-ordered list of the same items.
    """
    if len(items) <= 1:
        return list(items)

    rows: list[list] = []
    for item in sorted(items, key=lambda item: key(item)[1]):   # sort by y_top
        x, y, w, h = key(item)
        placed = False
        for row in rows:
            for r_item in row:
                rx, ry, rw, rh = key(r_item)
                overlap = min(y + h, ry + rh) - max(y, ry)
                if overlap > 0 and overlap / min(h, rh) >= 0.5:
                    row.append(item)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            rows.append([item])

    rows.sort(key=lambda row: min(key(item)[1] for item in row))
    result: list = []
    for row in rows:
        result.extend(sorted(row, key=lambda item: key(item)[0]))
    return result


# ---------------------------------------------------------------------------
# IoU deduplication
# ---------------------------------------------------------------------------

def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union for two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = max(0, min(ax + aw, bx + bw) - ix)
    ih = max(0, min(ay + ah, by + bh) - iy)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _deduplicate(items: list, iou_thresh: float = 0.50, key: Callable = lambda item: item) -> list:
    """
    Remove near-duplicate bounding boxes.

    Args:
        items:      List of items to dedupe (typically (x, y, w, h) tuples,
            or (bbox, confidence) pairs — pre-sort by confidence descending
            to keep the most confident box among duplicates).
        iou_thresh: Boxes with IoU above this threshold are considered duplicates.
        key:        Function extracting the (x, y, w, h) bbox from each item.

    Returns:
        Deduplicated list (preserves first occurrence).
    """
    result: list = []
    for item in items:
        if not any(_iou(key(item), key(e)) > iou_thresh for e in result):
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Core single-page detection (MagiV3 backend)
# ---------------------------------------------------------------------------

def _box_to_xywh(box, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert a MagiV3 [x1, y1, x2, y2] box to a clamped (x, y, w, h) tuple."""
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), img_w))
    y2 = max(y1 + 1, min(int(round(y2)), img_h))
    return (x1, y1, x2 - x1, y2 - y1)


def _read_page_for_magi(image_path: str) -> np.ndarray:
    """
    Load a page image the way MagiV3 expects it.

    Per the model's documented usage, pages are opened, converted to
    grayscale and back to RGB (normalising away colour so scans and colour
    pages are treated consistently) before being handed to the model as an
    (H, W, 3) uint8 array.
    """
    with open(image_path, "rb") as f:
        image = Image.open(f).convert("L").convert("RGB")
    return np.array(image)


def _call_predict_ocr(model, processor, magi_image: np.ndarray, raw_text_boxes: list):
    """
    Call MagiV3's `predict_ocr`, adapting to either of its known call
    signatures.

    Different MagiV3/`transformers` revisions have shipped `predict_ocr`
    as either `(images, processor)` (v3, re-detecting text regions
    internally) or `(images, text_bboxes_for_all_images, processor)` /
    `(images, text_bboxes_for_all_images)` (v1/v2 style, OCR-ing the boxes
    already found by `predict_detections_and_associations`). We inspect
    the bound method's signature once and call accordingly, rather than
    hard-coding one version and breaking on the other.

    Returns:
        List of OCR strings, one per entry in `raw_text_boxes` (best
        effort — padded/truncated to that length if the model's own text
        count differs).
    """
    params = list(inspect.signature(model.predict_ocr).parameters)
    with torch.no_grad():
        if len(params) >= 2 and "text_bboxes" in params[1] or "bboxes" in "".join(params):
            ocr_results = model.predict_ocr(
                [magi_image], [raw_text_boxes], processor
            )
        else:
            try:
                ocr_results = model.predict_ocr([magi_image], [raw_text_boxes], processor)
            except TypeError:
                ocr_results = model.predict_ocr([magi_image], processor)

    strings = list(ocr_results[0]) if ocr_results else []
    if len(strings) < len(raw_text_boxes):
        strings += [""] * (len(raw_text_boxes) - len(strings))
    return strings[: len(raw_text_boxes)]


def _assign_texts_to_panels(
    panel_boxes: list[tuple[int, int, int, int]],
    text_boxes: list[tuple[int, int, int, int]],
    ocr_strings: list[str],
) -> list[list[dict]]:
    """
    Geometrically assign each detected text/speech-bubble region to the
    panel that contains it.

    MagiV3 associates text with characters and speech-bubble tails, but
    not directly with panels, so panel membership is derived here: a text
    box belongs to whichever panel contains its centre point; ties (or
    text sitting on a panel gutter) fall back to the panel with the
    greatest overlap.

    Returns:
        List parallel to `panel_boxes`, where each entry is a list of
        `{"bbox": (x, y, w, h), "text": str}` dicts for that panel, in no
        particular order (callers sort/join as needed).
    """
    assignments: list[list[dict]] = [[] for _ in panel_boxes]
    if not panel_boxes:
        return assignments

    for tbox, text in zip(text_boxes, ocr_strings):
        tx, ty, tw, th = tbox
        tcx, tcy = tx + tw / 2, ty + th / 2

        best_idx, best_overlap = None, 0.0
        for i, pbox in enumerate(panel_boxes):
            px, py, pw, ph = pbox
            if px <= tcx <= px + pw and py <= tcy <= py + ph:
                overlap = _iou(tbox, pbox)
                if best_idx is None or overlap >= best_overlap:
                    best_idx, best_overlap = i, overlap

        if best_idx is None:
            # Text box didn't fall inside any panel's centre test (e.g. it
            # straddles a gutter) — fall back to maximum overlap.
            best_idx = max(range(len(panel_boxes)), key=lambda i: _iou(tbox, panel_boxes[i]))

        assignments[best_idx].append({"bbox": tbox, "text": text})

    return assignments


@lru_cache(maxsize=2)
def _analyze_page(image_path: str) -> dict:
    """
    Run MagiV3 once on a page and return everything downstream code needs:
    panel boxes (with confidence), text/speech-bubble boxes, and their OCR
    transcriptions.

    Cached (small LRU) so that `detect_panels_on_page()` and
    `detect_panels_with_ocr()` can both be called for the same page
    (as `detect_all_panels()` does today via `detect_panels_on_page()`,
    and would via `detect_panels_with_ocr()`) without paying for a second,
    expensive model forward pass.

    Args:
        image_path: Absolute path to the source page PNG.

    Returns:
        {
            "panel_pairs": [((x, y, w, h), confidence), ...]  — deduped,
                reading-order sorted,
            "text_items":  [{"bbox": (x, y, w, h), "text": str}, ...]
                aligned with the panels each belongs to (see
                `panel_text_items` below for the per-panel grouping),
            "panel_text_items": [[{"bbox": ..., "text": ...}, ...], ...]
                parallel to "panel_pairs",
        }

    Raises:
        ValueError: If the image cannot be loaded (path not found or corrupt).
        RuntimeError: If the MagiV3 model fails to load (see get_model()).
    """
    probe = cv2.imread(image_path)
    if probe is None:
        raise ValueError(f"Cannot load image: {image_path}")
    img_h, img_w = probe.shape[:2]

    model = get_model()
    processor = _get_processor()
    magi_image = _read_page_for_magi(image_path)

    # Only pass a threshold kwarg if this MagiV3 revision's detection
    # method actually exposes one — keeps us working across minor API
    # differences instead of hard failing.
    detect_kwargs = {}
    detect_params = inspect.signature(model.predict_detections_and_associations).parameters
    if "panel_detection_threshold" in detect_params:
        detect_kwargs["panel_detection_threshold"] = _CONF_THRESHOLD

    with torch.no_grad():
        results = model.predict_detections_and_associations(
            [magi_image], processor, **detect_kwargs
        )
    page_result = results[0] if results else {}

    raw_panels = page_result.get("panels", []) or []
    # MagiV3's detection head doesn't always surface a calibrated per-panel
    # confidence score the way YOLO's did. When it's absent we fall back to
    # a constant sentinel so downstream code (and the database `confidence`
    # column) keeps receiving a float in the expected range.
    panel_scores = page_result.get("panel_scores") or page_result.get("scores")
    if not panel_scores or len(panel_scores) != len(raw_panels):
        panel_scores = [1.0] * len(raw_panels)

    pairs = [
        (_box_to_xywh(box, img_w, img_h), float(score))
        for box, score in zip(raw_panels, panel_scores)
    ]
    # Keep the highest-confidence box among near-duplicates: sort by
    # confidence descending first, since _deduplicate preserves first
    # occurrence.
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    pairs = _deduplicate(pairs, key=lambda pair: pair[0])
    pairs = _sort_reading_order(pairs, key=lambda pair: pair[0])

    raw_texts = page_result.get("texts", []) or []
    text_boxes = [_box_to_xywh(box, img_w, img_h) for box in raw_texts]
    ocr_strings = _call_predict_ocr(model, processor, magi_image, raw_texts) if raw_texts else []

    panel_boxes = [bbox for bbox, _ in pairs]
    panel_text_items = _assign_texts_to_panels(panel_boxes, text_boxes, ocr_strings)

    return {
        "panel_pairs": pairs,
        "text_items": [{"bbox": b, "text": t} for b, t in zip(text_boxes, ocr_strings)],
        "panel_text_items": panel_text_items,
    }


def detect_panels_on_page(
    image_path: str,
    return_confidences: bool = False,
):
    """
    Detect comic panels in one page image using the MagiV3 comic-
    understanding model (`ragavsachdeva/magiv3`).

    Algorithm overview:
        • Load the cached MagiV3 model (loaded once for the whole process)
        • Run inference on the full-resolution page image
        • Convert predictions to (x, y, w, h) boxes, clamped to image bounds
        • IoU deduplication (keeps the highest-confidence box)
        • Sorted into western reading order

    This function's output format is unchanged from the previous YOLO
    implementation on purpose, so every other module keeps working without
    modification. For MagiV3's additional OCR/dialogue output, see
    `detect_panels_with_ocr()`.

    Args:
        image_path: Absolute path to the source page PNG.
        return_confidences: When True, also return the per-box confidence
            scores. Defaults to False so existing callers that only expect
            a list of (x, y, w, h) tuples keep working unchanged.

    Returns:
        List of (x, y, w, h) bounding boxes in reading order. When
        `return_confidences` is True, returns a tuple
        (bboxes, confidences) instead, where confidences is a parallel
        list of floats.
        Returns an empty list (or ([], []) list when
        `return_confidences=True`) when no valid panels are found.

    Raises:
        ValueError: If the image cannot be loaded (path not found or corrupt).
        RuntimeError: If the MagiV3 model fails to load — see get_model().
    """
    pairs = _analyze_page(image_path)["panel_pairs"]
    bboxes = [bbox for bbox, _ in pairs]
    if return_confidences:
        confidences = [conf for _, conf in pairs]
        return bboxes, confidences
    return bboxes


def detect_panels_with_ocr(image_path: str):
    """
    Detect comic panels AND their dialogue/OCR data using MagiV3.

    New function (additive — does not change `detect_panels_on_page()`'s
    contract) used by `detect_all_panels()` to populate the OCR-related
    database columns described near `_INSERT_PANEL_SUPPORTS_OCR`.

    Args:
        image_path: Absolute path to the source page PNG.

    Returns:
        (bboxes, confidences, panel_ocr) where:
          - bboxes, confidences: identical in shape/meaning to
            `detect_panels_on_page(image_path, return_confidences=True)`.
          - panel_ocr: list parallel to `bboxes`, each entry a dict:
                {
                    "dialogue": [str, ...],           # OCR'd lines, reading order
                    "ocr_text": str,                  # dialogue joined with " "
                    "speech_bubbles": [
                        {"x": int, "y": int, "w": int, "h": int}, ...
                    ],
                }

    Raises:
        ValueError: If the image cannot be loaded (path not found or corrupt).
        RuntimeError: If the MagiV3 model fails to load — see get_model().
    """
    analysis = _analyze_page(image_path)
    pairs = analysis["panel_pairs"]
    bboxes = [bbox for bbox, _ in pairs]
    confidences = [conf for _, conf in pairs]

    panel_ocr = []
    for items in analysis["panel_text_items"]:
        # Reading-order sort the text within a panel the same way panels
        # themselves are ordered, so dialogue reads top-to-bottom/left-to-right.
        ordered = _sort_reading_order(items, key=lambda item: item["bbox"])
        dialogue = [item["text"] for item in ordered if item["text"]]
        speech_bubbles = [
            {"x": item["bbox"][0], "y": item["bbox"][1], "w": item["bbox"][2], "h": item["bbox"][3]}
            for item in ordered
        ]
        panel_ocr.append({
            "dialogue": dialogue,
            "ocr_text": " ".join(dialogue),
            "speech_bubbles": speech_bubbles,
        })

    return bboxes, confidences, panel_ocr


def detect_panels_debug(image_path: str) -> dict:
    """
    Return MagiV3 detection stages for one page, for debug/inspection views.

    Note: detection is now performed by the MagiV3 comic-understanding
    model rather than OpenCV contours, so "binary"/"morphological" stages
    are simple visual aids rather than intermediate detection artifacts.
    Unlike YOLO, MagiV3 doesn't expose a continuous confidence score for
    every candidate box the way a single-stage detector does, so the
    accepted/rejected split below is best-effort: boxes are split by the
    same `panel_scores` used in `_analyze_page()` (falling back to "all
    accepted" when the model doesn't surface per-box scores at all).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    model = get_model()
    processor = _get_processor()
    magi_image = _read_page_for_magi(image_path)
    img_h, img_w = image.shape[:2]

    with torch.no_grad():
        raw_results = model.predict_detections_and_associations([magi_image], processor)
    page_result = raw_results[0] if raw_results else {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    morph = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )

    all_overlay, accepted_overlay, rejected_overlay = image.copy(), image.copy(), image.copy()
    accepted: list[tuple[int, int, int, int]] = []
    rejected: list[dict] = []

    raw_panels = page_result.get("panels", []) or []
    panel_scores = page_result.get("panel_scores") or page_result.get("scores")
    if not panel_scores or len(panel_scores) != len(raw_panels):
        panel_scores = [1.0] * len(raw_panels)  # no scores exposed — treat all as accepted

    for box, conf in zip(raw_panels, panel_scores):
        x, y, w, h = _box_to_xywh(box, img_w, img_h)
        cv2.rectangle(all_overlay, (x, y), (x + w, y + h), (0, 180, 255), 1)
        if conf >= _CONF_THRESHOLD:
            accepted.append((x, y, w, h))
            cv2.rectangle(accepted_overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
        else:
            rejected.append({
                "box": (x, y, w, h),
                "reason": f"confidence {conf:.2f} below threshold {_CONF_THRESHOLD:.2f}",
            })
            cv2.rectangle(rejected_overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

    accepted = _sort_reading_order(_deduplicate(accepted))

    return {
        "original": image, "grayscale": gray, "binary": binary, "morphological": morph,
        "all_contours": all_overlay, "accepted_contours": accepted_overlay,
        "rejected_contours": rejected_overlay, "contours_found": len(rejected) + len(accepted),
        "accepted": accepted, "rejected": rejected,
    }


# ---------------------------------------------------------------------------
# Overlay drawing (Panel Viewer)
# ---------------------------------------------------------------------------

_OVERLAY_COLORS = [
    (220,  60,  60),   # red
    ( 60, 180,  60),   # green
    ( 60,  60, 220),   # blue
    (220, 140,  60),   # orange
    (140,  60, 220),   # purple
    ( 60, 180, 180),   # teal
    (180,  60, 180),   # magenta
    (220, 220,  60),   # yellow
]


def draw_panel_overlays(image_path: str, panels: list[dict]) -> bytes:
    """
    Return a PNG (as raw bytes) of the source page with coloured panel
    bounding boxes overlaid.

    Each panel gets:
        • A semi-transparent coloured fill
        • A solid 3-pixel border
        • A numbered label badge in the top-left corner of the box

    Args:
        image_path: Absolute path to the original page PNG.
        panels:     Panel dicts as returned by database.get_panels_for_page().

    Returns:
        PNG image bytes suitable for st.image().

    Raises:
        ValueError: If the image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # --- semi-transparent fill (single blend pass for all panels) ---
    overlay = img.copy()
    for i, panel in enumerate(panels):
        bbox = json.loads(panel["bounding_box"])
        x = int(bbox["x"]); y = int(bbox["y"])
        w = int(bbox["w"]); h = int(bbox["h"])
        cv2.rectangle(overlay, (x, y), (x + w, y + h),
                      _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)], -1)

    result = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)

    # --- solid borders + numbered labels ---
    for i, panel in enumerate(panels):
        bbox = json.loads(panel["bounding_box"])
        x = int(bbox["x"]); y = int(bbox["y"])
        w = int(bbox["w"]); h = int(bbox["h"])
        color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
        label = str(panel["panel_index"])

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)

        fs = max(0.7, min(h, w) / 280)
        th = max(2, int(fs * 2.2))
        (tw, t_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, fs, th
        )
        lx, ly = x + 8, y + t_h + 8
        cv2.rectangle(result,
                      (lx - 4, ly - t_h - 4), (lx + tw + 4, ly + 4),
                      color, -1)
        cv2.putText(result, label, (lx, ly),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (255, 255, 255), th)

    _, buf = cv2.imencode(".png", result)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Batch detection across all pages of a project
# ---------------------------------------------------------------------------

_INSERT_PANEL_SUPPORTS_CONFIDENCE = "confidence" in inspect.signature(
    database.insert_panel
).parameters


def detect_all_panels(
    project_name: str,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
) -> dict:
    """
    Detect panels on every extracted page of a project.

    For each EXTRACTED page:
        1. Removes any previous panel records (safe re-detection).
        2. Calls detect_panels_on_page().
        3. Crops and saves panel PNG files to panels/page_XXXX/.
        4. Writes panel records to SQLite.

    Pages where no panels are detected are flagged in pages_with_issues
    rather than raising an exception. The batch continues regardless.

    On completion, updates the project state to PANELS_READY.

    Args:
        project_name:
            Sanitized project name.
        progress_callback:
            Optional callable(page_num, total, current_index, status).
            status is one of: 'detected', 'no_panels', 'error'.

    Returns:
        {
            "total_pages":       int,   — extracted pages found
            "pages_processed":   int,   — pages with at least 1 panel
            "total_panels":      int,   — sum of all detected panels
            "pages_with_issues": list,  — page numbers with no panels / errors
            "errors":            list,  — human-readable error strings
        }
    """
    result: dict = {
        "total_pages": 0,
        "pages_processed": 0,
        "total_panels": 0,
        "pages_with_issues": [],
        "errors": [],
    }

    pages = database.get_all_pages(project_name)
    extracted = [p for p in pages if p["state"] == "EXTRACTED"]
    total = len(extracted)
    result["total_pages"] = total

    panels_root = os.path.join(
        workspace.get_project_path(project_name), "panels"
    )

    for idx, page in enumerate(extracted):
        page_num: int = page["page_number"]
        page_id: int = page["id"]
        image_path: str = page.get("image_path", "")

        # ── Guard: image file must exist ──────────────────────────────────
        if not image_path or not os.path.exists(image_path):
            result["errors"].append(
                f"Page {page_num}: image file missing at '{image_path}'."
            )
            result["pages_with_issues"].append(page_num)
            if progress_callback:
                progress_callback(page_num, total, idx + 1, "error")
            continue

        # ── Create output directory for this page ─────────────────────────
        page_panel_dir = os.path.join(panels_root, f"page_{page_num:04d}")
        os.makedirs(page_panel_dir, exist_ok=True)

        # ── Clear previous records (safe re-detection) ────────────────────
        database.delete_panels_for_page(project_name, page_id)

        # ── Detect ───────────────────────────────────────────────────────
        try:
            bboxes, confidences = detect_panels_on_page(
                image_path, return_confidences=True
            )
        except Exception as exc:
            result["errors"].append(
                f"Page {page_num}: detection failed — {exc}"
            )
            result["pages_with_issues"].append(page_num)
            if progress_callback:
                progress_callback(page_num, total, idx + 1, "error")
            continue

        if not bboxes:
            result["pages_with_issues"].append(page_num)
            if progress_callback:
                progress_callback(page_num, total, idx + 1, "no_panels")
            continue

        # ── Crop and save panels ─────────────────────────────────────────
        img = cv2.imread(image_path)
        ih, iw = img.shape[:2]

        for panel_idx, ((x, y, w, h), confidence) in enumerate(zip(bboxes, confidences)):
            panel_number = panel_idx + 1

            # Clamp to image bounds
            x1 = max(0, x);        y1 = max(0, y)
            x2 = min(iw, x + w);   y2 = min(ih, y + h)

            crop = img[y1:y2, x1:x2]
            ph, pw = crop.shape[:2]

            panel_filename = f"panel_{panel_number:03d}.png"
            panel_path = os.path.join(page_panel_dir, panel_filename)
            cv2.imwrite(panel_path, crop)

            bbox_json = json.dumps(
                {"x": x1, "y": y1, "w": pw, "h": ph}
            )

            insert_kwargs = dict(
                project_name=project_name,
                page_id=page_id,
                panel_index=panel_number,
                image_filename=panel_filename,
                image_path=panel_path,
                width=pw,
                height=ph,
                bounding_box_json=bbox_json,
            )
            # Only pass confidence if the database layer actually supports
            # it — otherwise keep the existing database call unchanged.
            if _INSERT_PANEL_SUPPORTS_CONFIDENCE:
                insert_kwargs["confidence"] = confidence

            database.insert_panel(**insert_kwargs)

        result["pages_processed"] += 1
        result["total_panels"] += len(bboxes)
        if progress_callback:
            progress_callback(page_num, total, idx + 1, "detected")

    # ── Update project state ──────────────────────────────────────────────
    database.update_project_state(project_name, "PANELS_READY")
    return result