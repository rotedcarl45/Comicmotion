import sqlite3
import os
from core import config


def get_db_path(project_name: str) -> str:
    """
    Returns the absolute path to the SQLite database file for a given project.

    Args:
        project_name: The name of the project (used as folder name in workspaces).

    Returns:
        Absolute path string to the project's .db file.
    """
    workspace_root = config.get_workspace_root()
    db_filename = config.get("database_filename")
    return os.path.join(workspace_root, project_name, db_filename)


def initialize_database(project_name: str) -> None:
    """
    Creates the SQLite database and all required tables for a new project.
    This function is idempotent — calling it on an existing database is safe.

    Tables created:
        - projects:  Stores the single project record and its pipeline state.
        - pages:     One row per extracted comic page (populated in Module 2).
        - panels:    One row per detected panel (populated in Module 2).

    Args:
        project_name: The name of the project. Used to locate the database file.
    """
    db_path = get_db_path(project_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT    NOT NULL UNIQUE,
            pdf_filename    TEXT    NOT NULL,
            file_type       TEXT    NOT NULL DEFAULT 'pdf',
            total_pages     INTEGER,
            state           TEXT    NOT NULL DEFAULT 'INITIALIZED',
            extracted_at    TEXT,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS pages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id      INTEGER NOT NULL,
            page_number     INTEGER NOT NULL,
            image_filename  TEXT,
            image_path      TEXT,
            width           INTEGER,
            height          INTEGER,
            extracted_at    TEXT,
            state           TEXT    NOT NULL DEFAULT 'PENDING',
            FOREIGN KEY (project_id) REFERENCES projects(id)
        );

        CREATE TABLE IF NOT EXISTS panels (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id         INTEGER NOT NULL,
            panel_index     INTEGER NOT NULL,
            image_filename  TEXT,
            image_path      TEXT,
            width           INTEGER,
            height          INTEGER,
            bounding_box    TEXT,
            analysis_json   TEXT,
            confidence_score REAL,
            reading_order   INTEGER,
            state           TEXT    NOT NULL DEFAULT 'PENDING',
            FOREIGN KEY (page_id) REFERENCES pages(id)
        );

        CREATE TABLE IF NOT EXISTS story_sequences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            sequence_index INTEGER NOT NULL,
            panel_id INTEGER NOT NULL UNIQUE,
            speaker TEXT, text TEXT, narration TEXT,
            emotion TEXT, camera_suggestion TEXT,
            duration_seconds REAL NOT NULL DEFAULT 3.0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            state TEXT NOT NULL DEFAULT 'READY',
            FOREIGN KEY (project_id) REFERENCES projects(id),
            FOREIGN KEY (panel_id) REFERENCES panels(id)
        );

        CREATE TABLE IF NOT EXISTS audio_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sequence_id INTEGER NOT NULL UNIQUE,
            audio_path TEXT NOT NULL, voice TEXT NOT NULL,
            duration_seconds REAL NOT NULL, text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (sequence_id) REFERENCES story_sequences(id)
        );

        CREATE TABLE IF NOT EXISTS render_clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sequence_id INTEGER NOT NULL UNIQUE,
            clip_path TEXT NOT NULL, duration_seconds REAL NOT NULL,
            state TEXT NOT NULL DEFAULT 'RENDERED', error_message TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (sequence_id) REFERENCES story_sequences(id)
        );

        CREATE TABLE IF NOT EXISTS render_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL, output_path TEXT,
            state TEXT NOT NULL DEFAULT 'PENDING', progress INTEGER NOT NULL DEFAULT 0,
            error_message TEXT, updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (project_id) REFERENCES projects(id)
        );
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Migration (existing databases created before Module 2)
# ---------------------------------------------------------------------------

def _add_column_if_missing(
    cursor: sqlite3.Cursor, table: str, column: str, col_type: str
) -> None:
    """
    Attempts to add a column to an existing table.
    Silently does nothing if the column already exists.

    Args:
        cursor:   An open SQLite cursor.
        table:    Table name to alter.
        column:   Name of the column to add.
        col_type: SQLite type string (e.g. 'INTEGER', 'TEXT').
    """
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        pass  # Column already exists — safe to ignore.


def run_migrations(project_name: str) -> None:
    """
    Applies non-destructive schema migrations to an existing database.
    Safe to call multiple times (idempotent).

    Adds any columns introduced after the initial Module 1 schema so that
    projects created before Module 2 continue to work correctly.

    Args:
        project_name: The name of the project whose database to migrate.
    """
    db_path = get_db_path(project_name)
    if not os.path.exists(db_path):
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Module 2 additions
    _add_column_if_missing(cursor, "projects", "total_pages",  "INTEGER")
    _add_column_if_missing(cursor, "pages",    "image_path",   "TEXT")
    _add_column_if_missing(cursor, "pages",    "width",        "INTEGER")
    _add_column_if_missing(cursor, "pages",    "height",       "INTEGER")
    _add_column_if_missing(cursor, "pages",    "extracted_at", "TEXT")

    # Module 2.1 additions
    _add_column_if_missing(cursor, "projects", "file_type",    "TEXT")
    _add_column_if_missing(cursor, "projects", "extracted_at", "TEXT")

    # Module 3 additions
    _add_column_if_missing(cursor, "panels",   "image_path",   "TEXT")
    _add_column_if_missing(cursor, "panels",   "width",        "INTEGER")
    _add_column_if_missing(cursor, "panels",   "height",       "INTEGER")
    _add_column_if_missing(cursor, "panels",   "confidence_score", "REAL")
    _add_column_if_missing(cursor, "panels",   "reading_order", "INTEGER")

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS story_sequences (
            id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL,
            sequence_index INTEGER NOT NULL, panel_id INTEGER NOT NULL UNIQUE,
            speaker TEXT, text TEXT, narration TEXT, emotion TEXT, camera_suggestion TEXT,
            duration_seconds REAL NOT NULL DEFAULT 3.0, metadata_json TEXT NOT NULL DEFAULT '{}',
            state TEXT NOT NULL DEFAULT 'READY'
        );
        CREATE TABLE IF NOT EXISTS audio_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sequence_id INTEGER NOT NULL UNIQUE,
            audio_path TEXT NOT NULL, voice TEXT NOT NULL, duration_seconds REAL NOT NULL,
            text TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS render_clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sequence_id INTEGER NOT NULL UNIQUE,
            clip_path TEXT NOT NULL, duration_seconds REAL NOT NULL,
            state TEXT NOT NULL DEFAULT 'RENDERED', error_message TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS render_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL,
            output_path TEXT, state TEXT NOT NULL DEFAULT 'PENDING', progress INTEGER NOT NULL DEFAULT 0,
            error_message TEXT, updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)

    conn.commit()
    conn.close()


def insert_project(
    project_name: str,
    pdf_filename: str,
    file_type: str = "pdf",
) -> int:
    """
    Inserts the initial project record into the projects table.

    Args:
        project_name:  The sanitized project name (unique).
        pdf_filename:  The original filename of the uploaded comic file.
        file_type:     Format of the source file: 'pdf', 'cbz', or 'cbr'.
                       Defaults to 'pdf' for backward compatibility with Module 1.

    Returns:
        The integer row ID of the newly inserted project record.

    Raises:
        sqlite3.IntegrityError: If a project with this name already exists.
    """
    db_path = get_db_path(project_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO projects (name, pdf_filename, file_type, state) "
        "VALUES (?, ?, ?, 'INITIALIZED')",
        (project_name, pdf_filename, file_type),
    )
    project_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return project_id


def get_project(project_name: str) -> dict | None:
    """
    Retrieves the project record for a given project name.

    Args:
        project_name: The name of the project to query.

    Returns:
        A dictionary with project fields, or None if no project exists yet.
    """
    db_path = get_db_path(project_name)
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM projects WHERE name = ?", (project_name,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# Project state management (Module 2+)
# ---------------------------------------------------------------------------

def update_project_state(project_name: str, state: str) -> None:
    """
    Updates the pipeline state of a project.
    When state is 'EXTRACTED', also records the extraction timestamp.

    Valid states (by convention):
        INITIALIZED, EXTRACTED, ANALYZED, AUDIO_GENERATED,
        RENDERED, COMPLETED, ERROR

    Args:
        project_name: The sanitized project name.
        state:        The new state string.
    """
    db_path = get_db_path(project_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if state == "EXTRACTED":
        cursor.execute(
            "UPDATE projects "
            "SET state = ?, extracted_at = datetime('now'), updated_at = datetime('now') "
            "WHERE name = ?",
            (state, project_name),
        )
    else:
        cursor.execute(
            "UPDATE projects SET state = ?, updated_at = datetime('now') WHERE name = ?",
            (state, project_name),
        )
    conn.commit()
    conn.close()


def update_project_total_pages(project_name: str, total_pages: int) -> None:
    """
    Persists the total page count discovered during PDF extraction.

    Args:
        project_name: The sanitized project name.
        total_pages:  Number of pages in the PDF.
    """
    db_path = get_db_path(project_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE projects SET total_pages = ?, updated_at = datetime('now') WHERE name = ?",
        (total_pages, project_name),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Page record management (Module 2+)
# ---------------------------------------------------------------------------

def insert_page(
    project_name: str,
    project_id: int,
    page_number: int,
    image_filename: str,
    image_path: str,
    width: int,
    height: int,
    extracted_at: str,
) -> int:
    """
    Inserts a new page record after successful extraction.

    Args:
        project_name:   Sanitized project name — used to locate the correct DB.
        project_id:     Foreign key to the projects table.
        page_number:    1-based page number.
        image_filename: PNG filename only (e.g. 'page_0001.png').
        image_path:     Absolute path to the saved PNG file.
        width:          Rendered image width in pixels.
        height:         Rendered image height in pixels.
        extracted_at:   UTC timestamp string of extraction.

    Returns:
        The row ID of the inserted page record.
    """
    db_path = get_db_path(project_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO pages
            (project_id, page_number, image_filename, image_path,
             width, height, extracted_at, state)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'EXTRACTED')
        """,
        (project_id, page_number, image_filename, image_path,
         width, height, extracted_at),
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_page(
    project_name: str, project_id: int, page_number: int
) -> dict | None:
    """
    Retrieves a single page record by project and page number.
    Used by the extractor to check whether a page was already processed
    (resumability check).

    Args:
        project_name: Sanitized project name — scopes the DB lookup correctly.
        project_id:   ID of the parent project.
        page_number:  1-based page number.

    Returns:
        A dictionary of page fields, or None if the page has no record yet.
    """
    db_path = get_db_path(project_name)
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM pages WHERE project_id = ? AND page_number = ?",
        (project_id, page_number),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row is not None else None


def update_page_extracted(
    project_name: str,
    page_id: int,
    width: int,
    height: int,
    image_path: str,
    image_filename: str,
    extracted_at: str,
) -> None:
    """
    Updates an existing page record to EXTRACTED state with full metadata.
    Used when a page row already exists (e.g. from a previous failed attempt)
    and is being successfully re-processed.

    Args:
        project_name:   Sanitized project name — used to locate the correct DB.
        page_id:        Primary key of the page row to update.
        width:          Rendered image width in pixels.
        height:         Rendered image height in pixels.
        image_path:     Absolute path to the saved PNG file.
        image_filename: PNG filename only.
        extracted_at:   UTC timestamp string of extraction.
    """
    db_path = get_db_path(project_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE pages
        SET width = ?, height = ?, image_path = ?, image_filename = ?,
            extracted_at = ?, state = 'EXTRACTED'
        WHERE id = ?
        """,
        (width, height, image_path, image_filename, extracted_at, page_id),
    )
    conn.commit()
    conn.close()


def get_all_pages(project_name: str) -> list[dict]:
    """
    Returns all page records for a project, ordered by page number.

    Args:
        project_name: The sanitized project name.

    Returns:
        A list of page dictionaries. Empty list if no pages exist yet.
    """
    db_path = get_db_path(project_name)
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM pages ORDER BY page_number ASC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Panel record management (Module 3+)
# ---------------------------------------------------------------------------

def insert_panel(
    project_name: str,
    page_id: int,
    panel_index: int,
    image_filename: str,
    image_path: str,
    width: int,
    height: int,
    bounding_box_json: str,
    state: str = "DETECTED",
    confidence_score: float | None = None,
    reading_order: int | None = None,
) -> int:
    """
    Inserts a panel record after successful panel detection.

    Args:
        project_name:      Sanitized project name — locates the correct DB.
        page_id:           FK to the pages table.
        panel_index:       1-based reading-order index within the page.
        image_filename:    PNG filename (e.g. 'panel_001.png').
        image_path:        Absolute path to the cropped panel PNG.
        width:             Panel image width in pixels.
        height:            Panel image height in pixels.
        bounding_box_json: JSON string '{"x":N,"y":N,"w":N,"h":N}' on source page.
        state:             Initial state string. Defaults to 'DETECTED'.

    Returns:
        The row ID of the inserted panel record.
    """
    db_path = get_db_path(project_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO panels
            (page_id, panel_index, image_filename, image_path,
             width, height, bounding_box, state, confidence_score, reading_order)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (page_id, panel_index, image_filename, image_path,
         width, height, bounding_box_json, state, confidence_score,
         reading_order if reading_order is not None else panel_index),
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_panels_for_page(project_name: str, page_id: int) -> list[dict]:
    """
    Returns all panel records for a page, ordered by panel_index.

    Args:
        project_name: Sanitized project name.
        page_id:      Primary key of the parent page row.

    Returns:
        List of panel dicts, or empty list if none exist.
    """
    db_path = get_db_path(project_name)
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM panels WHERE page_id = ? ORDER BY panel_index ASC",
        (page_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_page_by_number(project_name: str, page_number: int) -> dict | None:
    """
    Returns a single page record looked up by its 1-based page number.

    Args:
        project_name: Sanitized project name.
        page_number:  1-based page number.

    Returns:
        A page dictionary, or None if not found.
    """
    db_path = get_db_path(project_name)
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM pages WHERE page_number = ?",
        (page_number,),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_panels_for_page(project_name: str, page_id: int) -> None:
    """
    Deletes all panel records for a given page.
    Used before re-detection to ensure a clean slate.

    Args:
        project_name: Sanitized project name.
        page_id:      Primary key of the parent page row.
    """
    db_path = get_db_path(project_name)
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM panels WHERE page_id = ?", (page_id,))
    conn.commit()
    conn.close()


def count_panels(project_name: str) -> int:
    """
    Returns the total number of panel records stored for a project.

    Args:
        project_name: Sanitized project name.

    Returns:
        Integer count, 0 if the DB does not exist.
    """
    db_path = get_db_path(project_name)
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM panels")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_all_panels(project_name: str) -> list[dict]:
    """Return panels in page and within-page reading order."""
    conn = sqlite3.connect(get_db_path(project_name)); conn.row_factory = sqlite3.Row
    rows = conn.execute("""SELECT panels.*, pages.page_number FROM panels
        JOIN pages ON pages.id = panels.page_id ORDER BY pages.page_number, panels.panel_index""").fetchall()
    conn.close(); return [dict(row) for row in rows]


def replace_story_sequences(project_name: str, sequences: list[dict]) -> None:
    """Atomically replace the derived story sequence for a project."""
    project = get_project(project_name)
    if project is None: raise ValueError(f"Unknown project: {project_name}")
    conn = sqlite3.connect(get_db_path(project_name))
    try:
        conn.execute("DELETE FROM audio_assets WHERE sequence_id IN (SELECT id FROM story_sequences WHERE project_id = ?)", (project['id'],))
        conn.execute("DELETE FROM render_clips WHERE sequence_id IN (SELECT id FROM story_sequences WHERE project_id = ?)", (project['id'],))
        conn.execute("DELETE FROM story_sequences WHERE project_id = ?", (project['id'],))
        conn.executemany("""INSERT INTO story_sequences
        (project_id,sequence_index,panel_id,speaker,text,narration,emotion,camera_suggestion,duration_seconds,metadata_json)
        VALUES (:project_id,:sequence_index,:panel_id,:speaker,:text,:narration,:emotion,:camera_suggestion,:duration_seconds,:metadata_json)""",
        [{**item, 'project_id': project['id']} for item in sequences])
        conn.commit()
    finally: conn.close()


def get_story_sequences(project_name: str) -> list[dict]:
    """Return story items with panel paths in playback order."""
    conn = sqlite3.connect(get_db_path(project_name)); conn.row_factory = sqlite3.Row
    rows = conn.execute("""SELECT s.*, p.image_path AS panel_path, p.page_id, a.audio_path, a.duration_seconds AS audio_duration
      FROM story_sequences s JOIN panels p ON p.id=s.panel_id LEFT JOIN audio_assets a ON a.sequence_id=s.id
      JOIN projects pr ON pr.id=s.project_id WHERE pr.name=? ORDER BY s.sequence_index""", (project_name,)).fetchall()
    conn.close(); return [dict(row) for row in rows]


def upsert_audio_asset(project_name: str, sequence_id: int, path: str, voice: str, duration: float, text: str) -> None:
    conn = sqlite3.connect(get_db_path(project_name)); conn.execute("""INSERT INTO audio_assets(sequence_id,audio_path,voice,duration_seconds,text)
      VALUES(?,?,?,?,?) ON CONFLICT(sequence_id) DO UPDATE SET audio_path=excluded.audio_path,voice=excluded.voice,duration_seconds=excluded.duration_seconds,text=excluded.text""", (sequence_id,path,voice,duration,text)); conn.commit(); conn.close()


def upsert_render_clip(project_name: str, sequence_id: int, path: str, duration: float, state: str = 'RENDERED', error: str | None = None) -> None:
    conn = sqlite3.connect(get_db_path(project_name)); conn.execute("""INSERT INTO render_clips(sequence_id,clip_path,duration_seconds,state,error_message)
      VALUES(?,?,?,?,?) ON CONFLICT(sequence_id) DO UPDATE SET clip_path=excluded.clip_path,duration_seconds=excluded.duration_seconds,state=excluded.state,error_message=excluded.error_message""", (sequence_id,path,duration,state,error)); conn.commit(); conn.close()


def get_render_clips(project_name: str) -> list[dict]:
    conn=sqlite3.connect(get_db_path(project_name)); conn.row_factory=sqlite3.Row
    rows=conn.execute("""SELECT c.*, s.sequence_index FROM render_clips c JOIN story_sequences s ON s.id=c.sequence_id
    JOIN projects p ON p.id=s.project_id WHERE p.name=? ORDER BY s.sequence_index""",(project_name,)).fetchall(); conn.close(); return [dict(r) for r in rows]


def create_render_job(project_name: str, output_path: str) -> int:
    project=get_project(project_name)
    if project is None: raise ValueError(f"Unknown project: {project_name}")
    conn=sqlite3.connect(get_db_path(project_name)); cur=conn.execute("INSERT INTO render_jobs(project_id,output_path,state) VALUES(?,?,'RUNNING')",(project['id'],output_path)); conn.commit(); conn.close(); return cur.lastrowid


def update_render_job(project_name: str, job_id: int, state: str, progress: int, error: str | None = None) -> None:
    conn=sqlite3.connect(get_db_path(project_name)); conn.execute("UPDATE render_jobs SET state=?,progress=?,error_message=?,updated_at=datetime('now') WHERE id=?",(state,progress,error,job_id)); conn.commit(); conn.close()
