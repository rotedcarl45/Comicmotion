import os

# Supported comic file extensions mapped to their format identifiers.
SUPPORTED_FORMATS: dict[str, str] = {
    ".pdf": "pdf",
    ".cbz": "cbz",
    ".cbr": "cbr",
}

# Magic bytes used to verify real file content regardless of extension.
_MAGIC: dict[str, bytes] = {
    "pdf": b"%PDF",
    "cbz": b"PK\x03\x04",  # ZIP local file header
    "cbr": b"Rar!",         # RAR signature
}


def detect_format(filename: str) -> str | None:
    """
    Returns the comic format identifier for a given filename based on extension.

    Args:
        filename: Original filename (e.g. 'MyComic.cbz').

    Returns:
        One of 'pdf', 'cbz', 'cbr', or None if the extension is unsupported.
    """
    ext = os.path.splitext(filename)[1].lower()
    return SUPPORTED_FORMATS.get(ext)


def validate_comic_file(uploaded_file) -> tuple[bool, str, str | None]:
    """
    Validates a Streamlit UploadedFile as a supported comic archive.

    Performs two checks per format:
        1. Extension must be one of .pdf, .cbz, .cbr.
        2. File header (magic bytes) must match the declared format.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        A tuple of (is_valid: bool, message: str, file_format: str | None).
        - is_valid:    True if the file passes both checks.
        - message:     Empty string on success; explanation of failure on error.
        - file_format: 'pdf', 'cbz', 'cbr', or None on failure.
    """
    filename = uploaded_file.name
    fmt = detect_format(filename)

    if fmt is None:
        supported = ", ".join(SUPPORTED_FORMATS.keys())
        return (
            False,
            f"'{filename}' is not a supported format. Accepted: {supported}",
            None,
        )

    # Read the first 4 bytes for magic-byte check, then reset stream.
    header = uploaded_file.read(4)
    uploaded_file.seek(0)

    expected_magic = _MAGIC[fmt]
    if not header.startswith(expected_magic):
        return (
            False,
            f"'{filename}' does not appear to be a valid {fmt.upper()} file "
            f"(invalid file header).",
            None,
        )

    return True, "", fmt


def validate_pdf(uploaded_file) -> tuple[bool, str]:
    """
    Backward-compatible wrapper for Module 1.
    Delegates to validate_comic_file and returns only the (bool, str) tuple.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        (is_valid: bool, message: str)
    """
    is_valid, message, _ = validate_comic_file(uploaded_file)
    return is_valid, message
