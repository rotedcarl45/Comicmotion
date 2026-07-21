"""
Universal comic ingestion router.

Supports PDF, CBZ (ZIP-based), and CBR (RAR-based) archives.

After ingestion every format produces an identical workspace layout:
    workspace/<project>/images/page_0001.png
                                page_0002.png
                                ...

The remaining pipeline (panel detection, Gemini, rendering) is entirely
format-agnostic — it always reads from images/ regardless of source type.
"""
import os
import re
import zipfile
from datetime import datetime, timezone
from typing import Callable, Optional

from PIL import Image

from core import database, workspace

# ---------------------------------------------------------------------------
# Image extensions considered valid comic pages inside archives.
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}


def _natural_sort_key(s: str):
    """
    Key function for natural (human-readable) sorting.
    Ensures '10.jpg' sorts after '9.jpg', not before '1.jpg'.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", s)
    ]


def _is_image(filename: str) -> bool:
    """Returns True if the filename has a supported image extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS


def _save_image_as_png(
    source_bytes: bytes, dest_path: str
) -> tuple[int, int]:
    """
    Writes image bytes to dest_path as a PNG file.
    Converts the source to PNG regardless of original format (JPEG, GIF, etc).

    Args:
        source_bytes: Raw bytes of the source image.
        dest_path:    Absolute path for the output PNG file.

    Returns:
        (width, height) of the saved image in pixels.
    """
    from io import BytesIO

    img = Image.open(BytesIO(source_bytes))
    # Convert palette/RGBA modes that don't save cleanly to PNG
    if img.mode in ("P", "RGBA"):
        img = img.convert("RGBA")
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(dest_path, format="PNG", optimize=False)
    return img.width, img.height


# ---------------------------------------------------------------------------
# CBZ
# ---------------------------------------------------------------------------

def _ingest_cbz(
    project_name: str,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
) -> dict:
    """
    Extracts pages from a CBZ (ZIP) archive.

    Finds the CBZ file in input_pdf/, reads its image members in natural
    filename order, converts each to PNG, and writes them to images/.
    Updates the SQLite database with page metadata.

    Args:
        project_name:      Sanitized project name.
        progress_callback: Optional callable (current, total, page_num, status).

    Returns:
        Result dict: {total, extracted, skipped, errors}
    """
    result = {"total": 0, "extracted": 0, "skipped": 0, "errors": []}

    input_dir = os.path.join(workspace.get_project_path(project_name), "input_pdf")
    cbz_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".cbz")]
    if not cbz_files:
        raise FileNotFoundError(f"No CBZ file found in '{input_dir}'.")
    cbz_path = os.path.join(input_dir, cbz_files[0])

    images_dir = os.path.join(workspace.get_project_path(project_name), "images")
    project = database.get_project(project_name)
    project_id: int = project["id"]

    with zipfile.ZipFile(cbz_path, "r") as zf:
        # Collect image members only, sorted in natural order
        image_names = sorted(
            [n for n in zf.namelist() if _is_image(os.path.basename(n))],
            key=_natural_sort_key,
        )

        total = len(image_names)
        result["total"] = total
        database.update_project_total_pages(project_name, total)

        for idx, member_name in enumerate(image_names):
            page_number = idx + 1
            filename = f"page_{page_number:04d}.png"
            dest_path = os.path.join(images_dir, filename)

            # Resumability check
            existing = database.get_page(project_name, project_id, page_number)
            if existing and existing["state"] == "EXTRACTED":
                result["skipped"] += 1
                if progress_callback:
                    progress_callback(page_number, total, page_number, "skipped")
                continue

            try:
                raw = zf.read(member_name)
                width, height = _save_image_as_png(raw, dest_path)
                extracted_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

                if existing:
                    database.update_page_extracted(
                        project_name, existing["id"], width, height,
                        dest_path, filename, extracted_at,
                    )
                else:
                    database.insert_page(
                        project_name, project_id, page_number, filename,
                        dest_path, width, height, extracted_at,
                    )

                result["extracted"] += 1
                if progress_callback:
                    progress_callback(page_number, total, page_number, "extracted")

            except Exception as exc:
                result["errors"].append({"page": page_number, "error": str(exc)})
                if progress_callback:
                    progress_callback(page_number, total, page_number, "error")

    return result


# ---------------------------------------------------------------------------
# CBR
# ---------------------------------------------------------------------------

def _ingest_cbr(
    project_name: str,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
) -> dict:
    """
    Extracts pages from a CBR (RAR) archive.

    Requires an external unrar binary (unrar, WinRAR, or 7-Zip) to be
    accessible on the system PATH. If no unrar tool is found, raises a
    RuntimeError with installation instructions.

    Args:
        project_name:      Sanitized project name.
        progress_callback: Optional callable (current, total, page_num, status).

    Returns:
        Result dict: {total, extracted, skipped, errors}
    """
    try:
        import rarfile
    except ImportError:
        raise RuntimeError(
            "The 'rarfile' library is not installed. "
            "Run: pip install rarfile"
        )

    result = {"total": 0, "extracted": 0, "skipped": 0, "errors": []}

    input_dir = os.path.join(workspace.get_project_path(project_name), "input_pdf")
    cbr_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".cbr")]
    if not cbr_files:
        raise FileNotFoundError(f"No CBR file found in '{input_dir}'.")
    cbr_path = os.path.join(input_dir, cbr_files[0])

    images_dir = os.path.join(workspace.get_project_path(project_name), "images")
    project = database.get_project(project_name)
    project_id: int = project["id"]

    try:
        rf = rarfile.RarFile(cbr_path)
    except rarfile.RarCannotExec:
        raise RuntimeError(
            "No unrar tool found on your system PATH. "
            "To enable CBR support, install one of the following:\n"
            "  - unrar (https://www.rarlab.com/rar_add.htm)\n"
            "  - 7-Zip (https://www.7-zip.org/) — add to PATH\n"
            "  - WinRAR — ensure it is on your PATH"
        )

    with rf:
        image_names = sorted(
            [n for n in rf.namelist() if _is_image(os.path.basename(n))],
            key=_natural_sort_key,
        )

        total = len(image_names)
        result["total"] = total
        database.update_project_total_pages(project_name, total)

        for idx, member_name in enumerate(image_names):
            page_number = idx + 1
            filename = f"page_{page_number:04d}.png"
            dest_path = os.path.join(images_dir, filename)

            existing = database.get_page(project_name, project_id, page_number)
            if existing and existing["state"] == "EXTRACTED":
                result["skipped"] += 1
                if progress_callback:
                    progress_callback(page_number, total, page_number, "skipped")
                continue

            try:
                raw = rf.read(member_name)
                width, height = _save_image_as_png(raw, dest_path)
                extracted_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

                if existing:
                    database.update_page_extracted(
                        project_name, existing["id"], width, height,
                        dest_path, filename, extracted_at,
                    )
                else:
                    database.insert_page(
                        project_name, project_id, page_number, filename,
                        dest_path, width, height, extracted_at,
                    )

                result["extracted"] += 1
                if progress_callback:
                    progress_callback(page_number, total, page_number, "extracted")

            except Exception as exc:
                result["errors"].append({"page": page_number, "error": str(exc)})
                if progress_callback:
                    progress_callback(page_number, total, page_number, "error")

    return result


# ---------------------------------------------------------------------------
# Public router
# ---------------------------------------------------------------------------

def ingest(
    project_name: str,
    file_format: str,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
) -> dict:
    """
    Routes extraction to the correct backend based on file_format.

    The PDF path delegates entirely to the existing pdf_extractor module
    so its behaviour is 100% unchanged. CBZ and CBR are handled here.

    After ingestion the caller must set project state to 'EXTRACTED' on
    full success. This is done inside each sub-extractor and in pdf_extractor.

    Args:
        project_name:      Sanitized project name.
        file_format:       One of 'pdf', 'cbz', 'cbr'.
        progress_callback: Optional callable (current, total, page_num, status).

    Returns:
        Result dict: {total, extracted, skipped, errors}

    Raises:
        ValueError: If file_format is not recognised.
    """
    from services import pdf_extractor  # local import avoids circular dependency

    if file_format == "pdf":
        return pdf_extractor.extract_pages(project_name, progress_callback)
    elif file_format == "cbz":
        result = _ingest_cbz(project_name, progress_callback)
    elif file_format == "cbr":
        result = _ingest_cbr(project_name, progress_callback)
    else:
        raise ValueError(
            f"Unknown file format '{file_format}'. Expected: pdf, cbz, cbr."
        )

    # Mark project EXTRACTED when fully complete (pdf_extractor does this itself)
    fully_done = (
        len(result["errors"]) == 0
        and (result["extracted"] + result["skipped"]) == result["total"]
    )
    if fully_done:
        database.update_project_state(project_name, "EXTRACTED")

    return result
