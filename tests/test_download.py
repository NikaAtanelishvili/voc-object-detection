"""download.py: MD5 verification, skip-if-exists, fallback order, and extraction, all offline.

file:// URLs let urllib "download" from tmp_path, so these tests never touch the network.
"""

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from vocdet.data import download as dl
from vocdet.data.download import (
    ArchiveSpec,
    DownloadError,
    ensure_archive,
    extract,
    md5_of,
    voc2012_spec,
)

BASE_DIR = "VOCdevkit/VOC2012"


def make_tar(path: Path) -> Path:
    """A tiny tar that mimics the VOC layout: VOCdevkit/VOC2012/ImageSets/Main/train.txt."""
    data = b"2008_000008\n"
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo(f"{BASE_DIR}/ImageSets/Main/train.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, str]:
    """A fake archive on 'the server' and its MD5."""
    (tmp_path / "server").mkdir()
    src = make_tar(tmp_path / "server" / "voc.tar")
    return src, hashlib.md5(src.read_bytes()).hexdigest()


def spec_for(urls: list[str], md5: str) -> ArchiveSpec:
    return ArchiveSpec(urls=urls, filename="voc.tar", md5=md5, base_dir=BASE_DIR)


def test_md5_of_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello voc")
    assert md5_of(f) == hashlib.md5(b"hello voc").hexdigest()


def test_download_verifies_md5(source: tuple[Path, str], tmp_path: Path) -> None:
    src, md5 = source
    archive = ensure_archive(spec_for([src.as_uri()], md5), tmp_path / "dl")
    assert archive.read_bytes() == src.read_bytes()
    assert not archive.with_name("voc.tar.part").exists()


def test_wrong_md5_raises_and_leaves_no_file(source: tuple[Path, str], tmp_path: Path) -> None:
    src, _ = source
    with pytest.raises(DownloadError, match="MD5"):
        ensure_archive(spec_for([src.as_uri()], "0" * 32), tmp_path / "dl")
    assert not (tmp_path / "dl" / "voc.tar").exists()


def test_verified_cached_archive_is_not_downloaded_again(
    source: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src, md5 = source
    ensure_archive(spec_for([src.as_uri()], md5), tmp_path / "dl")

    def fail(*_args: object) -> None:
        raise AssertionError("download_file must not be called for a verified cached archive")

    monkeypatch.setattr(dl, "download_file", fail)
    assert ensure_archive(spec_for([src.as_uri()], md5), tmp_path / "dl").exists()


def test_corrupt_cached_archive_is_replaced(source: tuple[Path, str], tmp_path: Path) -> None:
    src, md5 = source
    cached = tmp_path / "dl" / "voc.tar"
    cached.parent.mkdir()
    cached.write_bytes(b"truncated")
    archive = ensure_archive(spec_for([src.as_uri()], md5), tmp_path / "dl")
    assert md5_of(archive) == md5


def test_falls_back_when_primary_is_unreachable(source: tuple[Path, str], tmp_path: Path) -> None:
    src, md5 = source
    missing = (tmp_path / "server" / "does-not-exist.tar").as_uri()
    archive = ensure_archive(spec_for([missing, src.as_uri()], md5), tmp_path / "dl")
    assert md5_of(archive) == md5


def test_falls_back_when_primary_has_wrong_content(
    source: tuple[Path, str], tmp_path: Path
) -> None:
    src, md5 = source
    wrong = tmp_path / "server" / "wrong.tar"
    wrong.write_bytes(b"not the archive")
    archive = ensure_archive(spec_for([wrong.as_uri(), src.as_uri()], md5), tmp_path / "dl")
    assert md5_of(archive) == md5


def test_extract_then_skip(source: tuple[Path, str], tmp_path: Path) -> None:
    src, _ = source
    voc_dir = extract(src, tmp_path / "data", BASE_DIR)
    assert (voc_dir / "ImageSets" / "Main" / "train.txt").read_text() == "2008_000008\n"
    assert (voc_dir / dl.EXTRACTED_MARKER).exists()

    src.unlink()  # a second call must not even open the archive
    assert extract(src, tmp_path / "data", BASE_DIR) == voc_dir


def test_extract_rejects_archive_without_base_dir(tmp_path: Path) -> None:
    src = make_tar(tmp_path / "voc.tar")
    with pytest.raises(FileNotFoundError, match="VOC2007"):
        extract(src, tmp_path / "data", "VOCdevkit/VOC2007")


def test_voc2012_spec_reads_torchvision() -> None:
    from torchvision.datasets.voc import DATASET_YEAR_DICT

    spec = voc2012_spec()
    assert spec.md5 == DATASET_YEAR_DICT["2012"]["md5"]
    assert spec.urls[0] == DATASET_YEAR_DICT["2012"]["url"]
    assert spec.urls[-1] == dl.FALLBACK_URL
    assert spec.filename == "VOCtrainval_11-May-2012.tar"
