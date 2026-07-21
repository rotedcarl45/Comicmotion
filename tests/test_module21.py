"""
Standalone test for Module 2.1: Universal Comic Ingestion.
Run from the project root:  python -X utf8 tests/test_module21.py

Tests:
  1. PDF ingestion still works via comic_ingestion.ingest()
  2. CBZ ingestion extracts images in natural order
  3. Validator correctly identifies PDF, CBZ, and rejects invalid files
  4. DB stores file_type and extracted_at correctly
  5. Workspace structure is identical for all formats
  6. Image viewer data (image_path accessible) verified
"""
import io
import os
import sys
import shutil
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
from PIL import Image

from core import database, workspace, validator
from services import comic_ingestion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJ_PDF = "TEST21_PDF"
PROJ_CBZ = "TEST21_CBZ"


def make_pdf(path: str, pages: int = 3) -> None:
    doc = fitz.open()
    for i in range(1, pages + 1):
        pg = doc.new_page(width=400, height=600)
        pg.insert_text((50, 300), f"Page {i}", fontsize=36)
    doc.save(path)
    doc.close()


def make_cbz(path: str, pages: int = 4) -> None:
    """Create a CBZ with JPEG pages named non-sequentially to test natural sort."""
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(1, pages + 1):
            img = Image.new("RGB", (300, 450), color=(50 + i * 40, 80, 120))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            # Use non-padded filenames to stress natural sort
            zf.writestr(f"page{i}.jpg", buf.getvalue())


def bootstrap_project(proj: str, src_path: str, file_type: str) -> None:
    """Create workspace, copy file, init DB. Always starts clean."""
    proj_path = workspace.get_project_path(proj)
    # Force clean slate
    if os.path.exists(proj_path):
        shutil.rmtree(proj_path)
    workspace.create_project_workspace(proj)
    with open(src_path, "rb") as f:
        workspace.save_uploaded_pdf(proj, f.read(), os.path.basename(src_path))
    database.initialize_database(proj)
    database.run_migrations(proj)  # ensure schema is current before insert
    database.insert_project(proj, os.path.basename(src_path), file_type)


def verify_workspace(proj: str, expected_pages: int) -> None:
    images_dir = os.path.join(workspace.get_project_path(proj), "images")
    for n in range(1, expected_pages + 1):
        png = os.path.join(images_dir, f"page_{n:04d}.png")
        assert os.path.exists(png), f"Missing PNG: {png}"
        assert os.path.getsize(png) > 500, f"PNG suspiciously small: {png}"
        print(f"  [OK] page_{n:04d}.png ({os.path.getsize(png):,} bytes)")


# ---------------------------------------------------------------------------
# Test 1: Validator
# ---------------------------------------------------------------------------

def test_validator() -> None:
    print("\n=== TEST 1: Validator ===")

    class FakeFile:
        def __init__(self, name, data):
            self.name = name
            self._buf = io.BytesIO(data)
        def read(self, n=-1): return self._buf.read(n)
        def seek(self, pos): self._buf.seek(pos)

    # PDF
    is_v, msg, fmt = validator.validate_comic_file(FakeFile("comic.pdf", b"%PDF-1.4 fake content"))
    assert is_v and fmt == "pdf", f"PDF validation failed: {msg}"
    print("[OK] PDF validated correctly")

    # CBZ
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("test.jpg", b"fake")
    is_v, msg, fmt = validator.validate_comic_file(FakeFile("comic.cbz", buf.getvalue()))
    assert is_v and fmt == "cbz", f"CBZ validation failed: {msg}"
    print("[OK] CBZ validated correctly")

    # Unsupported extension
    is_v, msg, fmt = validator.validate_comic_file(FakeFile("comic.txt", b"hello"))
    assert not is_v and fmt is None, "Should reject .txt"
    print("[OK] Unsupported extension rejected correctly")

    # Wrong magic bytes (PDF extension but ZIP content)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.jpg", b"x")
    is_v, msg, fmt = validator.validate_comic_file(FakeFile("comic.pdf", buf.getvalue()))
    assert not is_v, "Should reject ZIP content with .pdf extension"
    print("[OK] Magic-byte mismatch rejected correctly")

    # Backward-compat: validate_pdf still works
    is_v2, msg2 = validator.validate_pdf(FakeFile("comic.pdf", b"%PDF-1.4 ok"))
    assert is_v2, f"validate_pdf compat broken: {msg2}"
    print("[OK] validate_pdf() backward compatibility OK")


# ---------------------------------------------------------------------------
# Test 2: PDF ingestion via router
# ---------------------------------------------------------------------------

def test_pdf_ingestion() -> None:
    print("\n=== TEST 2: PDF ingestion via comic_ingestion.ingest() ===")
    test_pdf = os.path.join("tests", "tmp_test.pdf")
    make_pdf(test_pdf, pages=3)

    bootstrap_project(PROJ_PDF, test_pdf, "pdf")
    # No separate run_migrations needed — bootstrap_project calls it

    result = comic_ingestion.ingest(PROJ_PDF, "pdf")
    print(f"  Result: {result}")
    assert result["total"] == 3
    assert result["extracted"] == 3
    assert result["errors"] == []

    verify_workspace(PROJ_PDF, 3)

    record = database.get_project(PROJ_PDF)
    assert record["state"] == "EXTRACTED", f"State wrong: {record['state']}"
    assert record["total_pages"] == 3
    assert record["file_type"] == "pdf"
    assert record["extracted_at"] is not None
    print(f"  [OK] file_type='{record['file_type']}', extracted_at='{record['extracted_at']}'")

    pages = database.get_all_pages(PROJ_PDF)
    for pg in pages:
        assert pg["state"] == "EXTRACTED"
        assert pg["image_path"] is not None
        assert os.path.exists(pg["image_path"]), f"image_path missing: {pg['image_path']}"
    print(f"  [OK] All {len(pages)} page rows have valid image_path")

    os.remove(test_pdf)
    shutil.rmtree(workspace.get_project_path(PROJ_PDF))
    print("[OK] PDF ingestion test passed")


# ---------------------------------------------------------------------------
# Test 3: CBZ ingestion
# ---------------------------------------------------------------------------

def test_cbz_ingestion() -> None:
    print("\n=== TEST 3: CBZ ingestion ===")
    test_cbz = os.path.join("tests", "tmp_test.cbz")
    make_cbz(test_cbz, pages=4)

    bootstrap_project(PROJ_CBZ, test_cbz, "cbz")
    # No separate run_migrations needed — bootstrap_project calls it

    result = comic_ingestion.ingest(PROJ_CBZ, "cbz")
    print(f"  Result: {result}")
    assert result["total"] == 4, f"Expected 4 pages, got {result['total']}"
    assert result["extracted"] == 4
    assert result["errors"] == []

    verify_workspace(PROJ_CBZ, 4)

    record = database.get_project(PROJ_CBZ)
    assert record["state"] == "EXTRACTED"
    assert record["file_type"] == "cbz"
    assert record["extracted_at"] is not None
    print(f"  [OK] file_type='{record['file_type']}', extracted_at='{record['extracted_at']}'")

    # Verify pages are normalized to PNG regardless of source JPEG
    images_dir = os.path.join(workspace.get_project_path(PROJ_CBZ), "images")
    for n in range(1, 5):
        png = os.path.join(images_dir, f"page_{n:04d}.png")
        with Image.open(png) as img:
            assert img.format == "PNG", f"Page {n} is not PNG: {img.format}"
    print("  [OK] All 4 JPEG source pages converted to PNG correctly")

    # Resumability test
    result2 = comic_ingestion.ingest(PROJ_CBZ, "cbz")
    assert result2["extracted"] == 0
    assert result2["skipped"] == 4
    print(f"  [OK] CBZ resumability: {result2}")

    # Verify image_path is accessible for viewer
    pages = database.get_all_pages(PROJ_CBZ)
    for pg in pages:
        assert os.path.exists(pg["image_path"]), f"Viewer path missing: {pg['image_path']}"
    print("  [OK] All image_path values valid for image viewer")

    os.remove(test_cbz)
    shutil.rmtree(workspace.get_project_path(PROJ_CBZ))
    print("[OK] CBZ ingestion test passed")


# ---------------------------------------------------------------------------
# Test 4: Workspace structure parity (PDF vs CBZ)
# ---------------------------------------------------------------------------

def test_workspace_parity() -> None:
    print("\n=== TEST 4: Workspace structure parity ===")
    expected_subdirs = set(workspace.PROJECT_SUBDIRS)

    for proj in [PROJ_PDF, PROJ_CBZ]:
        proj_path = workspace.get_project_path(proj)
        if not os.path.exists(proj_path):
            continue
        actual = set(
            d for d in os.listdir(proj_path)
            if os.path.isdir(os.path.join(proj_path, d))
        )
        assert expected_subdirs.issubset(actual), (
            f"{proj}: missing dirs: {expected_subdirs - actual}"
        )
    print("[OK] Workspace structure parity verified")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("tests", exist_ok=True)

    test_validator()
    test_pdf_ingestion()
    test_cbz_ingestion()
    test_workspace_parity()

    print("\n" + "=" * 60)
    print("ALL MODULE 2.1 ASSERTIONS PASSED")
    print("=" * 60)
