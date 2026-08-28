#!/usr/bin/env python3
"""
Run or resume the opt-in modeling-improvement agent on an existing QSAR run.

Requires ``outputs/<run_id>/final_report/`` from the deterministic workflow.

Examples:
  python scripts/run_modeling_agent.py outputs/<run_id>
  python scripts/run_modeling_agent.py outputs/<run_id> --resume
  python scripts/run_modeling_agent.py outputs/<run_id> --approve-exclusion pending
  python scripts/run_modeling_agent.py outputs/<run_id> --reject-exclusion pending
  python scripts/run_modeling_agent.py outputs/<run_id> --no-openai
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _prefer_environment_native_libs() -> None:
    """Re-exec so micromamba/conda libstdc++ is found before the system copy (WSL)."""
    lib = Path(sys.prefix) / "lib"
    if not (lib / "libstdc++.so.6").exists():
        return
    marker = str(lib.resolve())
    parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    if parts and str(Path(parts[0]).resolve()) == marker:
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([marker, *parts])
    os.execv(sys.executable, [sys.executable, *sys.argv])


_prefer_environment_native_libs()

from qsar_agent.agentic.runner import run_modeling_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", help="Path to outputs/<run_id>")
    parser.add_argument("--run-dir", dest="run_dir_opt", help="Path to outputs/<run_id>")
    parser.add_argument("--resume", action="store_true", help="Resume from the SQLite checkpointer")
    parser.add_argument("--approve-exclusion", metavar="ID", help="Resume with exclusion approval")
    parser.add_argument("--reject-exclusion", metavar="ID", help="Resume with exclusion rejection")
    parser.add_argument("--no-openai", action="store_true", help="Use deterministic fallback decisions")
    args = parser.parse_args()
    run_dir = Path(args.run_dir_opt or args.run_dir or "")
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    approval = None
    resume = bool(args.resume)
    if args.approve_exclusion:
        approval = {"approved": True, "proposal_id": args.approve_exclusion}
        resume = True
    elif args.reject_exclusion:
        approval = {"approved": False, "proposal_id": args.reject_exclusion}
        resume = True
    state = run_modeling_agent(
        run_dir,
        resume=resume,
        approval=approval,
        use_openai=not args.no_openai,
    )
    print(f"phase={state.phase} stopping_reason={state.stopping_reason} report={state.report_path}")


if __name__ == "__main__":
    main()
