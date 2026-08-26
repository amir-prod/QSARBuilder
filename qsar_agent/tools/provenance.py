"""Git commit and package-version provenance for the modeling handoff."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired, run

from qsar_agent.schemas.handoff import GitProvenance

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PACKAGE_IMPORT_NAMES = (
    ("scikit-learn", "sklearn"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("rdkit", "rdkit"),
    ("umap-learn", "umap"),
    ("deap", "deap"),
    ("mlxtend", "mlxtend"),
    ("joblib", "joblib"),
    ("streamlit", "streamlit"),
    ("mordred", "mordred"),
)


def get_git_provenance(cwd: Path | None = None) -> GitProvenance:
    """Return HEAD commit and dirty flag, or an unavailable record with a reason."""
    root = Path(cwd) if cwd is not None else _PROJECT_ROOT
    try:
        commit_proc = run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if commit_proc.returncode != 0:
            reason = (commit_proc.stderr or commit_proc.stdout or "git rev-parse failed").strip()
            return GitProvenance(available=False, reason=reason or "not a git repository")
        commit = commit_proc.stdout.strip()
        status_proc = run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        dirty = bool((status_proc.stdout or "").strip()) if status_proc.returncode == 0 else False
        return GitProvenance(available=True, commit=commit, dirty=dirty)
    except (FileNotFoundError, TimeoutExpired, CalledProcessError, OSError) as exc:
        return GitProvenance(available=False, reason=str(exc) or "git unavailable")


def _distribution_version(dist_name: str, module_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        pass
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    return "unavailable"


def collect_package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "qsar_agent": _qsar_agent_version(),
    }
    for dist_name, module_name in _PACKAGE_IMPORT_NAMES:
        versions[dist_name] = _distribution_version(dist_name, module_name)
    return versions


def _qsar_agent_version() -> str:
    try:
        from qsar_agent import __version__

        return str(__version__)
    except Exception:
        return "unavailable"
