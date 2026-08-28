"""Artifact and run-directory management."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

SAFE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def generate_run_id() -> str:
    return uuid.uuid4().hex[:12]


def sanitize_filename(name: str) -> str:
    base = Path(name).name
    if not base or ".." in base or "/" in base or "\\" in base:
        raise ValueError("Unsafe filename")
    return base


def get_run_dir(output_dir: str | Path, run_id: str) -> Path:
    if not SAFE_NAME_PATTERN.match(run_id):
        raise ValueError("Invalid run ID")
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_input_dataset(source: str | Path, run_dir: Path) -> Path:
    dest = run_dir / "input_dataset.csv"
    source_path = Path(source).resolve()
    dest_path = dest.resolve()
    if source_path == dest_path:
        return dest_path
    shutil.copy2(source_path, dest_path)
    return dest_path


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_sorted_ids(ids: list[str] | tuple[str, ...]) -> str:
    """SHA-256 of sorted unique compound IDs, one per line."""
    payload = "\n".join(sorted({str(i) for i in ids}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` via a same-directory temp file and ``os.replace``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=str(out.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, out)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    payload = json.dumps(data, indent=2, default=str) + "\n"
    atomic_write_text(path, payload)


def create_zip_archive(run_dir: Path, run_id: str) -> Path:
    zip_path = run_dir / f"qsar_agent_run_{run_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(run_dir.rglob("*")):
            if file_path.is_file() and file_path != zip_path:
                arcname = file_path.relative_to(run_dir)
                zf.write(file_path, arcname)
    return zip_path


def validate_artifact_exists(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Expected artifact not found: {p}")
    return p
