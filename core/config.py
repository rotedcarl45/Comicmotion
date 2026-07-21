import json
import os

_config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")

with open(_config_path, "r") as f:
    _config = json.load(f)


def get(key: str):
    """
    Retrieves a top-level value from config.json.

    Args:
        key: The key to retrieve from the config dictionary.

    Returns:
        The value for the given key.

    Raises:
        KeyError: If the key does not exist in the config.
    """
    return _config[key]


def get_workspace_root() -> str:
    """
    Returns the absolute path to the workspace root directory.
    This is always relative to the project root (where config.json lives),
    not the current working directory.

    Returns:
        Absolute path string to the workspaces directory.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, _config["workspace_root"])
