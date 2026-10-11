import os
from pathlib import Path

import pytest

from visual_service.validation.path_security import (
    ALLOWED_EXTENSIONS,
    InvalidMediaFile,
    MediaNotFound,
    PathOutsideMediaRoot,
    resolve_media_path,
)


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    root = tmp_path / "media"
    root.mkdir()
    return root


@pytest.fixture
def sample_video(media_root: Path) -> Path:
    video = media_root / "clip.mp4"
    video.write_bytes(b"\x00" * 64)
    return video


class TestValidPath:
    def test_returns_resolved_path(self, media_root: Path, sample_video: Path):
        result = resolve_media_path(sample_video, media_root)
        assert result == sample_video.resolve()

    def test_accepts_all_allowed_extensions(self, media_root: Path):
        for ext in ALLOWED_EXTENSIONS:
            video = media_root / f"test{ext}"
            video.write_bytes(b"\x00")
            result = resolve_media_path(video, media_root)
            assert result == video.resolve()

    def test_accepts_uppercase_extension(self, media_root: Path):
        video = media_root / "clip.MP4"
        video.write_bytes(b"\x00")
        result = resolve_media_path(video, media_root)
        assert result == video.resolve()

    def test_subdirectory_within_root(self, media_root: Path):
        sub = media_root / "sub" / "deep"
        sub.mkdir(parents=True)
        video = sub / "nested.mkv"
        video.write_bytes(b"\x00")
        result = resolve_media_path(video, media_root)
        assert result == video.resolve()


class TestPathTraversal:
    def test_dotdot_rejected(self, media_root: Path, tmp_path: Path):
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"\x00")
        traversal = media_root / ".." / "outside.mp4"
        with pytest.raises(PathOutsideMediaRoot):
            resolve_media_path(traversal, media_root)

    def test_absolute_path_outside_root(self, media_root: Path, tmp_path: Path):
        outside = tmp_path / "elsewhere.mp4"
        outside.write_bytes(b"\x00")
        with pytest.raises(PathOutsideMediaRoot):
            resolve_media_path(outside, media_root)


class TestSymlink:
    def test_symlink_outside_root_rejected(self, media_root: Path, tmp_path: Path):
        target = tmp_path / "secret.mp4"
        target.write_bytes(b"\x00")
        link = media_root / "sneaky.mp4"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("Symlink creation requires elevated privileges on this OS")
        with pytest.raises(PathOutsideMediaRoot):
            resolve_media_path(link, media_root)

    def test_symlink_within_root_accepted(self, media_root: Path, sample_video: Path):
        link = media_root / "alias.mp4"
        try:
            link.symlink_to(sample_video)
        except OSError:
            pytest.skip("Symlink creation requires elevated privileges on this OS")
        result = resolve_media_path(link, media_root)
        assert result == sample_video.resolve()


class TestUNCPaths:
    def test_backslash_unc_rejected(self, media_root: Path):
        unc = Path("\\\\server\\share\\video.mp4")
        with pytest.raises(PathOutsideMediaRoot, match="UNC"):
            resolve_media_path(unc, media_root)

    def test_forward_slash_unc_rejected(self, media_root: Path):
        unc = Path("//server/share/video.mp4")
        with pytest.raises(PathOutsideMediaRoot, match="UNC"):
            resolve_media_path(unc, media_root)


class TestNonExistentFile:
    def test_missing_file_raises(self, media_root: Path):
        missing = media_root / "does_not_exist.mp4"
        with pytest.raises(MediaNotFound):
            resolve_media_path(missing, media_root)


class TestDirectoryRejected:
    def test_directory_rejected(self, media_root: Path):
        subdir = media_root / "subdir"
        subdir.mkdir()
        with pytest.raises(InvalidMediaFile, match="regular file"):
            resolve_media_path(subdir, media_root)


class TestInvalidExtension:
    @pytest.mark.parametrize("filename", [
        "file.txt", "file.jpg", "file.exe", "file.py", "file.html",
    ])
    def test_disallowed_extension_rejected(self, media_root: Path, filename: str):
        bad = media_root / filename
        bad.write_bytes(b"\x00")
        with pytest.raises(InvalidMediaFile, match="Unsupported extension"):
            resolve_media_path(bad, media_root)

    def test_no_extension_rejected(self, media_root: Path):
        noext = media_root / "videofile"
        noext.write_bytes(b"\x00")
        with pytest.raises(InvalidMediaFile, match="Unsupported extension"):
            resolve_media_path(noext, media_root)
