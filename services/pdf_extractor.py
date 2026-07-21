import fitz  # PyMuPDF
import os
from datetime import datetime, timezone
from typing import Callable, Optional

from core import database, workspace

# Render scale applied to the native PDF coordinate space.
# 2.0 = 144 DPI effective output (native PDF is 72 DPI).
# This preserves sharp artwork without upscaling beyond the original intent.
RENDER_SCALE: float = 2.0


def get_pdf_path(project_name: str) -> str:
    """
    Locates the PDF file inside the project's input_pdf/ folder.

    Expects exactly one PDF file to be present (placed there during project
    creation in Module 1).

    Args:
        project_name: The sanitized project name.

    Returns:
        Absolute path string to the PDF file.

    Raises:
        FileNotFoundError: If no PDF is found in the input_pdf/ folder.
    """
    input_pdf_dir = os.path.join(
        workspace.get_project_path(project_name), "input_pdf"
    )
    pdf_files = [
        f for f in os.listdir(input_pdf_dir) if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF file found in '{input_pdf_dir}'. "
            "Re-create the project and upload the PDF again."
        )
    return os.path.join(input_pdf_dir, pdf_files[0])


def extract_pages(
    project_name: str,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
) -> dict:
    """
    Extracts every page from the project's PDF as a high-resolution PNG.

    Behaviour:
        - Pages already marked EXTRACTED in the database are skipped, making
          the function fully resumable after interruption.
        - Each page is rendered at RENDER_SCALE (2x = 144 DPI).
        - PNG files are saved sequentially: page_0001.png, page_0002.png, …
        - On full success, updates the project state to 'EXTRACTED'.

    Args:
        project_name:
            The sanitized project name. Used to locate the DB and workspace.
        progress_callback:
            Optional callable invoked after every page attempt.
            Signature: callback(current: int, total: int, page_num: int, status: str)
            where status is one of: "extracted", "skipped", "error".

    Returns:
        A summary dictionary:
            {
                "total":     int,   # total pages in the PDF
                "extracted": int,   # pages rendered and saved this run
                "skipped":   int,   # pages already extracted (resumed)
                "errors":    list   # list of {"page": int, "error": str} dicts
            }

    Raises:
        FileNotFoundError: If the PDF cannot be located.
        RuntimeError: If the project record is missing from the database.
    """
    pdf_path = get_pdf_path(project_name)
    images_dir = os.path.join(
        workspace.get_project_path(project_name), "images"
    )

    project = database.get_project(project_name)
    if project is None:
        raise RuntimeError(
            f"No database record found for project '{project_name}'. "
            "The database may be corrupted."
        )
    project_id: int = project["id"]

    result = {"total": 0, "extracted": 0, "skipped": 0, "errors": []}

    doc = fitz.open(pdf_path)
    total_pages: int = len(doc)
    result["total"] = total_pages

    # Persist total page count immediately so the UI can display it
    # even if extraction is interrupted.
    database.update_project_total_pages(project_name, total_pages)

    render_matrix = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)

    for page_index in range(total_pages):
        page_number: int = page_index + 1          # 1-based numbering
        filename: str = f"page_{page_number:04d}.png"
        image_path: str = os.path.join(images_dir, filename)

        # ── Resumability check ─────────────────────────────────────────────
        existing = database.get_page(project_name, project_id, page_number)
        if existing and existing["state"] == "EXTRACTED":
            result["skipped"] += 1
            if progress_callback:
                progress_callback(page_index + 1, total_pages, page_number, "skipped")
            continue

        # ── Render and save ────────────────────────────────────────────────
        try:
            page = doc[page_index]
            pixmap = page.get_pixmap(matrix=render_matrix)
            width: int = pixmap.width
            height: int = pixmap.height
            pixmap.save(image_path)

            extracted_at: str = (
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            )

            if existing:
                # Row exists but was previously in ERROR or PENDING — update it.
                database.update_page_extracted(
                    project_name=project_name,
                    page_id=existing["id"],
                    width=width,
                    height=height,
                    image_path=image_path,
                    image_filename=filename,
                    extracted_at=extracted_at,
                )
            else:
                database.insert_page(
                    project_name=project_name,
                    project_id=project_id,
                    page_number=page_number,
                    image_filename=filename,
                    image_path=image_path,
                    width=width,
                    height=height,
                    extracted_at=extracted_at,
                )

            result["extracted"] += 1
            if progress_callback:
                progress_callback(page_index + 1, total_pages, page_number, "extracted")

        except Exception as exc:
            result["errors"].append({"page": page_number, "error": str(exc)})
            if progress_callback:
                progress_callback(page_index + 1, total_pages, page_number, "error")

    doc.close()

    # ── Promote project state when fully complete ──────────────────────────
    fully_done = (
        len(result["errors"]) == 0
        and (result["extracted"] + result["skipped"]) == total_pages
    )
    if fully_done:
        database.update_project_state(project_name, "EXTRACTED")

    return result
