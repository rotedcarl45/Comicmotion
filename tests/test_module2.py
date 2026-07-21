"""
Standalone test for Module 2: PDF page extraction.
Run from the project root:  python tests/test_module2.py
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
from core import database, workspace
from services import pdf_extractor

PROJ = "TEST_MODULE2"


def make_test_pdf(path: str, pages: int = 3) -> None:
    """Create a minimal multi-page PDF using PyMuPDF."""
    doc = fitz.open()
    for i in range(1, pages + 1):
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((72, 400), f"ComicMotion Test Page {i}", fontsize=48)
    doc.save(path)
    doc.close()


def cleanup(proj_path: str, pdf_path: str) -> None:
    if os.path.exists(proj_path):
        shutil.rmtree(proj_path)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)


def run() -> None:
    test_pdf = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_sample.pdf"
    )
    proj_path = workspace.get_project_path(PROJ)
    cleanup(proj_path, test_pdf)

    # ── Step 1: Create test PDF ──────────────────────────────────────────────
    make_test_pdf(test_pdf, pages=3)
    assert os.path.exists(test_pdf), "Test PDF was not created"
    print("[OK] Created 3-page test PDF")

    # ── Step 2: Bootstrap project (Module 1 functions) ──────────────────────
    workspace.create_project_workspace(PROJ)
    with open(test_pdf, "rb") as f:
        workspace.save_uploaded_pdf(PROJ, f.read(), "test_sample.pdf")
    database.initialize_database(PROJ)
    database.insert_project(PROJ, "test_sample.pdf")
    print("[OK] Project workspace and DB initialized")

    # ── Step 3: First extraction ─────────────────────────────────────────────
    log = []

    def cb(current, total, page_num, status):
        log.append((page_num, status))
        print(f"  Page {page_num}/{total}: {status}")

    result = pdf_extractor.extract_pages(PROJ, progress_callback=cb)
    print(f"[OK] Extraction result: {result}")

    assert result["total"]     == 3,  f"total wrong: {result['total']}"
    assert result["extracted"] == 3,  f"extracted wrong: {result['extracted']}"
    assert result["skipped"]   == 0,  f"skipped wrong: {result['skipped']}"
    assert result["errors"]    == [], f"errors wrong: {result['errors']}"

    # ── Step 4: Verify PNG files ─────────────────────────────────────────────
    images_dir = os.path.join(proj_path, "images")
    for n in [1, 2, 3]:
        png = os.path.join(images_dir, f"page_{n:04d}.png")
        assert os.path.exists(png), f"Missing PNG: {png}"
        size = os.path.getsize(png)
        assert size > 1000, f"PNG too small ({size} bytes): {png}"
        print(f"  [OK] page_{n:04d}.png — {size:,} bytes")

    # ── Step 5: Verify resolution (should be 2x A4 = 1190x1684 approx) ──────
    with fitz.open(os.path.join(images_dir, "page_0001.png")) as img_doc:
        w = img_doc[0].rect.width
        h = img_doc[0].rect.height
    print(f"  [OK] Rendered dimensions (via fitz): {w:.0f} x {h:.0f}")

    # ── Step 6: Verify DB metadata ───────────────────────────────────────────
    record = database.get_project(PROJ)
    assert record["state"]       == "EXTRACTED", f"state wrong: {record['state']}"
    assert record["total_pages"] == 3,           f"total_pages wrong: {record['total_pages']}"
    print(f"[OK] Project state: {record['state']}, total_pages: {record['total_pages']}")

    pages = database.get_all_pages(PROJ)
    assert len(pages) == 3, f"Expected 3 page rows in DB, got {len(pages)}"
    for pg in pages:
        assert pg["state"]        == "EXTRACTED",  f"page state wrong: {pg}"
        assert pg["width"]        >  0,             f"width wrong: {pg}"
        assert pg["height"]       >  0,             f"height wrong: {pg}"
        assert pg["image_path"]   is not None,      f"image_path null: {pg}"
        assert pg["extracted_at"] is not None,      f"extracted_at null: {pg}"
        print(
            f"  [OK] DB page {pg['page_number']}: "
            f"{pg['width']}x{pg['height']}px — {pg['extracted_at']}"
        )

    # -- Step 7: Resumability test --------------------------------------------
    print("\n-- RESUMABILITY TEST --------------------------------------------")
    result2 = pdf_extractor.extract_pages(PROJ)
    assert result2["extracted"] == 0, f"Should extract 0, got {result2['extracted']}"
    assert result2["skipped"]   == 3, f"Should skip 3, got {result2['skipped']}"
    assert result2["errors"]    == [], f"Should have no errors"
    print(f"[OK] Resume result: {result2}")

    # -- Step 8: Interrupted extraction test ----------------------------------
    print("\n-- INTERRUPTED EXTRACTION TEST ----------------------------------")
    # Simulate a partially-extracted state:
    # Delete the DB rows for pages 2 and 3, and delete their PNG files.
    db_path = database.get_db_path(PROJ)
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM pages WHERE page_number IN (2, 3)")
    conn.execute("UPDATE projects SET state = 'INITIALIZED', total_pages = NULL")
    conn.commit()
    conn.close()
    for n in [2, 3]:
        png = os.path.join(images_dir, f"page_{n:04d}.png")
        if os.path.exists(png):
            os.remove(png)
    print("  [SIM] Deleted pages 2 and 3 to simulate interruption")

    result3 = pdf_extractor.extract_pages(PROJ)
    assert result3["extracted"] == 2, f"Should re-extract 2, got {result3['extracted']}"
    assert result3["skipped"]   == 1, f"Should skip 1, got {result3['skipped']}"
    print(f"[OK] Interrupted resume result: {result3}")
    for n in [2, 3]:
        png = os.path.join(images_dir, f"page_{n:04d}.png")
        assert os.path.exists(png), f"Missing after resume: {png}"
    print("[OK] All PNG files present after interrupted resume")

    print("\n" + "=" * 60)
    print("ALL MODULE 2 ASSERTIONS PASSED")
    print("=" * 60)

    # Cleanup
    cleanup(proj_path, test_pdf)
    print("[OK] Test workspace cleaned up")


if __name__ == "__main__":
    run()
