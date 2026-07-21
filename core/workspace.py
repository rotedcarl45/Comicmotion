import os
import re
import shutil
from core import config


# Sub-folders created inside every project workspace.
PROJECT_SUBDIRS = [
    "input_pdf",
    "images",
    "panels",
    "audio",
    "video_clips",
    "output",
    "story",
]


def sanitize_project_name(name: str) -> str:
    """
    Converts a raw user-entered project name into a safe folder name.
    Strips leading/trailing whitespace, replaces spaces with underscores,
    and removes any character that is not alphanumeric or underscore/hyphen.

    Args:
        name: Raw user input string.

    Returns:
        A sanitized string safe for use as a filesystem folder name.

    Raises:
        ValueError: If the sanitized name is empty.
    """
    sanitized = name.strip()
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = re.sub(r"[^\w\-]", "", sanitized)
    if not sanitized:
        raise ValueError("Project name cannot be empty or contain only special characters.")
    return sanitized


def get_project_path(project_name: str) -> str:
    """
    Returns the absolute path to a project's workspace folder.

    Args:
        project_name: The sanitized project name.

    Returns:
        Absolute path string to the project folder inside workspaces/.
    """
    return os.path.join(config.get_workspace_root(), project_name)


def project_exists(project_name: str) -> bool:
    """
    Checks whether a project workspace folder already exists on disk.

    Args:
        project_name: The sanitized project name.

    Returns:
        True if the folder exists, False otherwise.
    """
    return os.path.isdir(get_project_path(project_name))


def create_project_workspace(project_name: str) -> str:
    """
    Creates the full folder structure for a new project workspace.
    Creates the project root directory and all required sub-directories.

    Folder structure created:
        workspaces/
        └── <project_name>/
            ├── input_pdf/
            ├── images/
            ├── panels/
            ├── audio/
            ├── video_clips/
            └── output/

    Args:
        project_name: The sanitized project name.

    Returns:
        The absolute path to the created project root directory.

    Raises:
        FileExistsError: If the project folder already exists.
    """
    project_path = get_project_path(project_name)

    if os.path.exists(project_path):
        raise FileExistsError(f"Project '{project_name}' already exists at: {project_path}")

    for subdir in PROJECT_SUBDIRS:
        os.makedirs(os.path.join(project_path, subdir), exist_ok=True)

    return project_path


def save_uploaded_pdf(project_name: str, pdf_bytes: bytes, original_filename: str) -> str:
    """
    Saves the uploaded PDF file bytes into the project's input_pdf/ folder.

    Args:
        project_name:      The sanitized project name.
        pdf_bytes:         Raw bytes of the uploaded PDF file.
        original_filename: The original filename as provided by the user.

    Returns:
        The absolute path to the saved PDF file.
    """
    destination = os.path.join(
        get_project_path(project_name), "input_pdf", original_filename
    )
    with open(destination, "wb") as f:
        f.write(pdf_bytes)
    return destination


def list_projects() -> list[str]:
    """
    Returns a list of all existing project names by scanning the workspaces root.

    Returns:
        A list of folder names found in the workspaces directory.
        Returns an empty list if the workspaces directory does not exist yet.
    """
    workspace_root = config.get_workspace_root()
    if not os.path.isdir(workspace_root):
        return []
    return [
        d for d in os.listdir(workspace_root)
        if os.path.isdir(os.path.join(workspace_root, d))
    ]


def delete_project_workspace(project_name: str) -> None:
    """
    Permanently deletes the project's entire workspace folder, including:
        input_pdf/, images/, panels/, audio/, video_clips/, output/

    Handles read-only files on Windows by resetting their permissions before
    deletion (common when files were opened by another process and have been
    closed but remain marked read-only).

    Args:
        project_name: The sanitized project name.

    Raises:
        FileNotFoundError: If the workspace folder does not exist.
        PermissionError: If a file is locked and cannot be deleted.
    """
    import stat

    project_path = get_project_path(project_name)
    if not os.path.exists(project_path):
        raise FileNotFoundError(
            f"Workspace folder not found: {project_path}"
        )

    def _on_error(func, path, exc_info):
        """
        Error handler for shutil.rmtree.
        Attempts to clear the read-only flag on the file and retry.
        Re-raises if the retry also fails.
        """
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            raise exc_info[1]

    shutil.rmtree(project_path, onerror=_on_error)
