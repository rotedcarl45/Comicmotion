"""
Module 3 test: Panel Detection Engine
Run from project root: python -X utf8 tests/test_module3.py

Tests:
  1. detect_panels_on_page() returns correct panels on synthetic page (3-panel layout).
  2. Reading order is correct (top-left → top-right → bottom).
  3. Re-detection clears old panel records before inserting new ones.
  4. SQLite records are written correctly (page_id, panel_index, bbox, path, dimensions).
  5. Project state transitions to PANELS_READY.
  6. Panel image files exist on disk and have non-zero size.
  7. detect_all_panels() result summary is correct.
  8. draw_panel_overlays() produces non-empty PNG bytes.
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import database, workspace
from services import panel_detector

PROJECT = "TEST_MODULE3"


# ---------------------------------------------------------------------------
# Synthetic test page builder
# ---------------------------------------------------------------------------

def make_test_page(width: int = 900, height: int = 1200) -> np.ndarray:
    """
    Build a synthetic comic page:
        ┌────────────┬────────────┐
        │  Panel 1   │  Panel 2   │  (top row)
        │  top-left  │  top-right │
        ├────────────┴────────────┤
        │        Panel 3          │  (bottom row, full-width)
        └─────────────────────────┘

    Reading order: 1 (top-left) → 2 (top-right) → 3 (bottom)
    """
    page = np.ones((height, width, 3), dtype=np.uint8) * 255  # white

    # Draw panels as dark-bordered rectangles with slightly-off-white fill
    panels = [
        (30,  30,  380, 520),   # panel 1: top-left
        (490, 30,  380, 520),   # panel 2: top-right
        (30,  580, 840, 580),   # panel 3: bottom full-width
    ]
    for x, y, w, h in panels:
        # Fill (dark enough so threshold picks it up)
        cv2.rectangle(page, (x + 5, y + 5), (x + w - 5, y + h - 5),
                      (200, 200, 200), -1)
        # Black border
        cv2.rectangle(page, (x, y), (x + w, y + h), (0, 0, 0), 6)

    return page


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_project() -> None:
    proj_path = workspace.get_project_path(PROJECT)
    if os.path.exists(proj_path):
        shutil.rmtree(proj_path)
    workspace.create_project_workspace(PROJECT)
    database.initialize_database(PROJECT)
    database.insert_project(PROJECT, "test.pdf", "pdf")


def insert_fake_page(page_img: np.ndarray, page_number: int) -> int:
    """Save a synthetic page image and register it in the DB. Returns page_id."""
    images_dir = os.path.join(workspace.get_project_path(PROJECT), "images")
    filename = f"page_{page_number:04d}.png"
    img_path = os.path.join(images_dir, filename)
    cv2.imwrite(img_path, page_img)

    proj_rec = database.get_project(PROJECT)
    proj_id = proj_rec["id"]

    database.update_project_total_pages(PROJECT, page_number)
    database.insert_page(
        project_name=PROJECT,
        project_id=proj_id,
        page_number=page_number,
        image_filename=filename,
        image_path=img_path,
        width=page_img.shape[1],
        height=page_img.shape[0],
        extracted_at="2026-01-01 00:00:00",
    )
    database.update_project_state(PROJECT, "EXTRACTED")

    page_rec = database.get_page_by_number(PROJECT, page_number)
    return page_rec["id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_detection_and_reading_order() -> None:
    print("\n=== TEST 1: Panel Detection + Reading Order ===")

    page_img = make_test_page()
    tmp_path = os.path.join(workspace.get_project_path(PROJECT), "images", "tmp_test.png")
    cv2.imwrite(tmp_path, page_img)

    bboxes = panel_detector.detect_panels_on_page(tmp_path)
    os.remove(tmp_path)

    assert len(bboxes) == 3, f"Expected 3 panels, got {len(bboxes)}: {bboxes}"
    print(f"[OK] Detected 3 panels: {bboxes}")

    # Reading order: top-left → top-right → bottom
    # Panel 1 y ≈ 30, Panel 2 y ≈ 30 (same row), Panel 3 y ≈ 580
    p1, p2, p3 = bboxes
    assert p1[0] < p2[0], "Panel 1 must be left of Panel 2 (same row)"
    assert p1[1] < p3[1] and p2[1] < p3[1], "Top row must precede bottom panel"
    print("[OK] Reading order: top-left → top-right → bottom")


def test_batch_detection_and_db() -> None:
    print("\n=== TEST 2: Batch Detection + SQLite Records ===")

    # Two pages
    page1 = make_test_page()
    page2 = make_test_page(width=900, height=1200)

    page1_id = insert_fake_page(page1, 1)
    page2_id = insert_fake_page(page2, 2)

    result = panel_detector.detect_all_panels(PROJECT)
    print(f"  Result: {result}")

    assert result["total_pages"] == 2,  f"Expected 2 total pages, got {result['total_pages']}"
    assert result["pages_processed"] == 2, f"Expected 2 processed, got {result['pages_processed']}"
    assert result["total_panels"] == 6,  f"Expected 6 panels (3×2), got {result['total_panels']}"
    assert result["errors"] == [],      f"Unexpected errors: {result['errors']}"
    print(f"[OK] {result['total_panels']} panels across {result['pages_processed']} pages")

    panels_p1 = database.get_panels_for_page(PROJECT, page1_id)
    assert len(panels_p1) == 3, f"Expected 3 panels for page 1, got {len(panels_p1)}"
    print("[OK] 3 panel DB records for page 1")

    for panel in panels_p1:
        assert panel["panel_index"] in (1, 2, 3)
        assert panel["image_filename"].endswith(".png")
        assert os.path.exists(panel["image_path"]), f"Missing: {panel['image_path']}"
        assert panel["width"] > 0 and panel["height"] > 0
        bbox = json.loads(panel["bounding_box"])
        assert all(k in bbox for k in ("x", "y", "w", "h"))
        assert panel["state"] == "DETECTED"
        sz = os.path.getsize(panel["image_path"])
        assert sz > 0, f"Panel PNG is empty: {panel['image_path']}"
    print("[OK] All panel PNG files exist and are non-empty")
    print("[OK] All bounding boxes contain x,y,w,h")
    print("[OK] All panel states are DETECTED")


def test_project_state() -> None:
    print("\n=== TEST 3: Project State → PANELS_READY ===")
    rec = database.get_project(PROJECT)
    assert rec["state"] == "PANELS_READY", (
        f"Expected PANELS_READY, got {rec['state']}"
    )
    print("[OK] Project state is PANELS_READY")

    total = database.count_panels(PROJECT)
    assert total == 6, f"Expected 6 panels total, got {total}"
    print(f"[OK] count_panels() = {total}")


def test_redetection() -> None:
    print("\n=== TEST 4: Re-detection (clean slate) ===")
    # Run detection again — must not produce duplicates
    database.update_project_state(PROJECT, "EXTRACTED")
    result2 = panel_detector.detect_all_panels(PROJECT)
    total2 = database.count_panels(PROJECT)
    assert total2 == 6, f"Expected 6 after re-detection, got {total2}"
    print("[OK] Re-detection produced exactly 6 panels (no duplicates)")


def test_overlay() -> None:
    print("\n=== TEST 5: draw_panel_overlays() ===")
    pages = database.get_all_pages(PROJECT)
    p1 = pages[0]
    panels = database.get_panels_for_page(PROJECT, p1["id"])
    png_bytes = panel_detector.draw_panel_overlays(p1["image_path"], panels)
    assert isinstance(png_bytes, bytes) and len(png_bytes) > 1000
    print(f"[OK] Overlay PNG: {len(png_bytes):,} bytes")


def test_workspace_structure() -> None:
    print("\n=== TEST 6: Workspace structure ===")
    panels_dir = os.path.join(workspace.get_project_path(PROJECT), "panels")
    for page_num in (1, 2):
        page_dir = os.path.join(panels_dir, f"page_{page_num:04d}")
        assert os.path.isdir(page_dir), f"Missing: {page_dir}"
        pngs = [f for f in os.listdir(page_dir) if f.endswith(".png")]
        assert len(pngs) == 3, f"Expected 3 PNGs in {page_dir}, got {pngs}"
    print("[OK] panels/page_0001/ and panels/page_0002/ each contain 3 PNGs")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Setting up test project…")
    bootstrap_project()

    test_detection_and_reading_order()
    test_batch_detection_and_db()
    test_project_state()
    test_redetection()
    test_overlay()
    test_workspace_structure()

    # Cleanup
    workspace.delete_project_workspace(PROJECT)
    print("\n" + "=" * 60)
    print("ALL MODULE 3 ASSERTIONS PASSED")
    print("=" * 60)
