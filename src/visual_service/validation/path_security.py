from __future__ import annotations

from pathlib import Path

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".mkv", ".webm", ".mov", ".avi",
})


class PathSecurityError(Exception):
    pass


class PathOutsideMediaRoot(PathSecurityError):
    pass


class MediaNotFound(PathSecurityError):
    pass


class InvalidMediaFile(PathSecurityError):
    pass


def resolve_media_path(requested: Path, media_root: Path) -> Path:
    req_str = str(requested)
    if req_str.startswith("\\\\") or req_str.startswith("//"):
        raise PathOutsideMediaRoot(f"UNC/network paths are not allowed: {req_str}")

    resolved_root = media_root.resolve(strict=True)

    try:
        resolved = requested.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise MediaNotFound(f"File not found: {requested}") from exc

    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        raise PathOutsideMediaRoot(
            f"Path {resolved} is outside media root {resolved_root}"
        )

    if not resolved.is_file():
        raise InvalidMediaFile(f"Not a regular file: {resolved}")

    if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise InvalidMediaFile(
            f"Unsupported extension {resolved.suffix!r}. "
            f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    return resolved
