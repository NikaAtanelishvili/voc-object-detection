"""Download and extract the VOC2012 trainval tar, idempotently and with MD5 verification.

The URL, filename, and MD5 are read from torchvision at runtime (PLAN.md §3), not typed from memory.
The fallback mirror is a personal site (pjreddie.com), so the MD5 check, not the host, is what
guarantees we got the official archive (ADR-006).
"""

import hashlib
import logging
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

FALLBACK_URL = "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"
CHUNK_BYTES = 1 << 20  # 1 MiB
LOG_EVERY_BYTES = 100 << 20  # progress line every 100 MiB
EXTRACTED_MARKER = ".extracted_ok"


class ChecksumError(RuntimeError):
    """A downloaded or cached archive does not match the expected MD5."""


class DownloadError(RuntimeError):
    """No URL produced a file with the expected MD5."""


@dataclass(frozen=True)
class ArchiveSpec:
    """Where to get an archive and how to recognise the right one."""

    urls: list[str]
    filename: str
    md5: str
    base_dir: str  # folder inside the archive that holds the dataset, e.g. VOCdevkit/VOC2012


def voc2012_spec() -> ArchiveSpec:
    """Primary URL, filename, MD5, and base dir from torchvision; our fallback mirror appended."""
    from torchvision.datasets.voc import DATASET_YEAR_DICT  # slow import, only needed here

    info = DATASET_YEAR_DICT["2012"]
    return ArchiveSpec(
        urls=[info["url"], FALLBACK_URL],
        filename=info["filename"],
        md5=info["md5"],
        base_dir=info["base_dir"],
    )


def md5_of(path: Path) -> str:
    """MD5 hex digest, read in chunks so a 2 GB file never sits in memory."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "md5").hexdigest()


def download_file(url: str, dest: Path) -> None:
    """Stream url to dest, logging progress. Writes to dest.part first so a crash never leaves a
    truncated file under the final name (which a later run would mistake for a finished download).
    """
    part = dest.with_name(dest.name + ".part")
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_log = LOG_EVERY_BYTES
        while chunk := response.read(CHUNK_BYTES):
            out.write(chunk)
            done += len(chunk)
            if done >= next_log:
                pct = f" ({100 * done / total:.0f}%)" if total else ""
                logger.info("  %d MiB%s", done >> 20, pct)
                next_log += LOG_EVERY_BYTES
    part.replace(dest)


def ensure_archive(spec: ArchiveSpec, download_dir: Path) -> Path:
    """Return the path of a verified archive, downloading only if needed.

    A cached file with the right MD5 is reused; one with the wrong MD5 is deleted. URLs are tried in
    order, and a download that fails or has the wrong MD5 moves on to the next URL.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    archive = download_dir / spec.filename
    if archive.exists():
        if md5_of(archive) == spec.md5:
            logger.info("Archive already present and verified: %s", archive)
            return archive
        logger.warning("Cached %s has the wrong MD5; deleting it and downloading again", archive)
        archive.unlink()

    errors: list[str] = []
    for url in spec.urls:
        try:
            download_file(url, archive)
        except (urllib.error.URLError, OSError) as e:
            errors.append(f"{url}: {e}")
            logger.warning("Download failed from %s: %s", url, e)
            continue
        actual = md5_of(archive)
        if actual == spec.md5:
            logger.info("MD5 verified (%s) for archive from %s", actual, url)
            return archive
        errors.append(f"{url}: MD5 {actual} != expected {spec.md5}")
        logger.warning("MD5 mismatch for archive from %s; deleting it", url)
        archive.unlink()
    raise DownloadError(f"Could not get a verified {spec.filename}:\n" + "\n".join(errors))


def extract(archive: Path, extract_dir: Path, base_dir: str) -> Path:
    """Extract archive into extract_dir and return extract_dir / base_dir.

    A marker file is written only after extraction finishes, so an interrupted extraction is redone
    on the next run. filter="data" rejects absolute paths and links that escape extract_dir.
    """
    dataset_dir = extract_dir / base_dir
    marker = dataset_dir / EXTRACTED_MARKER
    if marker.exists():
        logger.info("Already extracted: %s", dataset_dir)
        return dataset_dir
    logger.info("Extracting %s to %s", archive, extract_dir)
    with tarfile.open(archive) as tar:
        tar.extractall(extract_dir, filter="data")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"{archive} did not contain the expected folder {base_dir}")
    marker.write_text(archive.name + "\n")
    return dataset_dir


def download_voc2012(root: Path) -> Path:
    """Download if needed, verify, and extract VOC2012 trainval; return root/VOCdevkit/VOC2012."""
    spec = voc2012_spec()
    archive = ensure_archive(spec, root / "downloads")
    return extract(archive, root, spec.base_dir)
