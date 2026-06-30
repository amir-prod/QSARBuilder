"""Artifact and run-directory management."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
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


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


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
