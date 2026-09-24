"""Command-line entry point for the agentic QSAR workflow.

    python -m qsar_agent.cli run --csv data.csv --smiles-col smiles --activity-col pIC50

Acceptance thresholds are not library defaults: the CLI asks what you are looking
for, showing a literature-typical suggestion for each criterion that you can
accept or override. Scripted runs supply the same values through flags,
``--criteria-file``, or ``--accept-suggested``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from qsar_agent.config import load_env_file
from qsar_agent.llm.provider import (
    LLMConfigurationError,
    get_llm_client,
    resolve_model_name,
    resolve_provider_name,
)
from qsar_agent.schemas.agentic import TaskDetection
from qsar_agent.services.agentic_loop import (
    AgenticConfig,
    AgenticQSARWorkflow,
)
from qsar_agent.tools.acceptance import (
    OPTIONAL_BY_DEFAULT,
    AcceptanceCriteria,
    CriterionSpec,
    spec_by_field,
    specs_for_task,
    suggested_value,
)
from qsar_agent.tools.task_detection import detect_task

RULE = "=" * 78


class CriteriaResolutionError(RuntimeError):
    """Raised when acceptance criteria cannot be determined."""


def _flag_to_dest(spec: CriterionSpec) -> str:
    return spec.cli_flag.lstrip("-").replace("-", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qsar-agent",
        description="Autonomous multi-agent QSAR modelling from a CSV of compounds.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Run the agentic workflow on a CSV dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run.add_argument("--csv", required=True, help="Input CSV containing compounds.")
    run.add_argument("--smiles-col", required=True, help="Column holding SMILES strings.")
    run.add_argument("--activity-col", required=True, help="Column holding the target activity.")
    run.add_argument("--id-col", default=None, help="Optional compound identifier column.")
    run.add_argument(
        "--task",
        choices=["auto", "regression", "classification"],
        default="auto",
        help="Force the modelling task instead of inferring it from the activity column.",
    )

    split = run.add_argument_group("data handling")
    split.add_argument(
        "--split",
        choices=["random", "stratified", "scaffold"],
        default="random",
        help="Train/test split strategy.",
    )
    split.add_argument("--test-size", type=float, default=0.2, help="Test-set fraction.")
    split.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds.")
    split.add_argument(
        "--scramble-repeats", type=int, default=10, help="y-randomisation shuffles."
    )
    split.add_argument("--ad-k", type=int, default=5, help="Neighbours for the k-NN domain.")
    split.add_argument(
        "--ad-z", type=float, default=3.0, help="Std-dev multiplier for the domain threshold."
    )
    split.add_argument("--seed", type=int, default=42, help="Random seed.")
    split.add_argument(
        "--min-compounds", type=int, default=20, help="Refuse to model fewer compounds than this."
    )

    budget = run.add_argument_group("iteration budget")
    budget.add_argument(
        "--max-iterations", type=int, default=6, help="Maximum agent iterations."
    )
    budget.add_argument(
        "--budget-seconds", type=float, default=900.0, help="Wall-clock budget for the loop."
    )
    budget.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for sklearn.")
    budget.add_argument("--output-dir", default="outputs", help="Where run artifacts are written.")
    budget.add_argument("--run-id", default=None, help="Reuse a specific run identifier.")

    llm = run.add_argument_group("language model")
    llm.add_argument(
        "--llm-provider",
        default=None,
        help="Provider name. 'mock' uses a scripted stub for offline demos and tests only.",
    )
    llm.add_argument("--llm-model", default=None, help="Model name to request from the provider.")

    criteria = run.add_argument_group(
        "acceptance criteria",
        description=(
            "Omit these to be asked interactively. Each flag disables the prompt for that "
            "criterion; pass 'off' to disable the criterion entirely."
        ),
    )
    for spec in _all_specs():
        criteria.add_argument(
            spec.cli_flag,
            dest=_flag_to_dest(spec),
            default=None,
            metavar="VALUE",
            help=f"{spec.label} (suggested {spec.suggested:g}; 'off' to disable). {spec.source}.",
        )
    criteria.add_argument(
        "--criteria-file",
        default=None,
        help="JSON file mapping criterion names to thresholds (or null to disable).",
    )
    criteria.add_argument(
        "--accept-suggested",
        action="store_true",
        help="Use the suggested values without prompting (non-interactive runs).",
    )
    criteria.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt; fail if the criteria are not fully specified.",
    )

    subparsers.add_parser(
        "show-criteria",
        help="Print the suggested acceptance criteria for each task and exit.",
    ).add_argument(
        "--task",
        choices=["regression", "classification"],
        default=None,
        help="Limit the listing to one task.",
    )

    return parser


def _all_specs() -> list[CriterionSpec]:
    from qsar_agent.tools.acceptance import CRITERION_SPECS

    return list(CRITERION_SPECS)


# -- criteria resolution -------------------------------------------------------


def _parse_threshold(raw: str, spec: CriterionSpec) -> float | None:
    text = raw.strip().lower()
    if text in {"off", "skip", "none", "disable", "disabled"}:
        return None
    try:
        return float(text)
    except ValueError:
        raise CriteriaResolutionError(
            f"Could not read a number for {spec.cli_flag}: {raw!r}. "
            "Give a number, or 'off' to disable this criterion."
        ) from None


def load_criteria_file(path: str | Path) -> dict[str, float | None]:
    """Read a JSON criteria file, validating every key against the known criteria."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CriteriaResolutionError(f"{path}: expected a JSON object of criterion thresholds.")
    resolved: dict[str, float | None] = {}
    for key, value in data.items():
        spec = _resolve_key(key)
        if value is None:
            resolved[spec.field] = None
        else:
            try:
                resolved[spec.field] = float(value)
            except (TypeError, ValueError):
                raise CriteriaResolutionError(
                    f"{path}: criterion {key!r} must be a number or null, got {value!r}."
                ) from None
    return resolved


def _resolve_key(key: str) -> CriterionSpec:
    normalised = key.strip().lstrip("-").replace("-", "_")
    try:
        return spec_by_field(normalised)
    except KeyError:
        pass
    for spec in _all_specs():
        if _flag_to_dest(spec) == normalised:
            return spec
    known = ", ".join(spec.field for spec in _all_specs())
    raise CriteriaResolutionError(f"Unknown acceptance criterion {key!r}. Known: {known}.")


def _explicit_flag_values(args: argparse.Namespace) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for spec in _all_specs():
        raw = getattr(args, _flag_to_dest(spec), None)
        if raw is not None:
            values[spec.field] = _parse_threshold(str(raw), spec)
    return values


def prompt_for_criteria(
    task: str,
    detection: TaskDetection,
    preset: dict[str, float | None],
    stream_in: TextIO,
    stream_out: TextIO,
) -> dict[str, float | None]:
    """Ask the user what they are looking for, one criterion at a time."""
    specs = specs_for_task(task)
    print(RULE, file=stream_out)
    print("What are you looking for in a validated model?", file=stream_out)
    print(RULE, file=stream_out)
    print(f"Detected task: {task} — {detection.reason}", file=stream_out)
    print(
        "\nFor each criterion: press Enter to accept the suggestion, type a number to use your "
        "own, or type 'off' to drop the criterion.\n",
        file=stream_out,
    )

    resolved: dict[str, float | None] = {}
    for spec in specs:
        if spec.field in preset:
            resolved[spec.field] = preset[spec.field]
            chosen = preset[spec.field]
            shown = "disabled" if chosen is None else f"{chosen:g}"
            print(f"  {spec.label}: {shown} (from the command line)", file=stream_out)
            continue

        suggestion = suggested_value(spec, task)
        optional = spec.field in OPTIONAL_BY_DEFAULT
        symbol = ">=" if spec.direction == "min" else "<="
        default_label = "off" if optional else f"{suggestion:g}"

        print(f"\n{spec.label} {symbol} ?", file=stream_out)
        print(f"  {spec.description}", file=stream_out)
        if optional:
            print(
                f"  No general value exists ({spec.source}); disabled unless you give one.",
                file=stream_out,
            )
        else:
            print(f"  Suggested: {suggestion:g} — {spec.source}", file=stream_out)
        stream_out.write(f"  Your requirement [{default_label}]: ")
        stream_out.flush()

        answer = stream_in.readline()
        if answer == "":  # EOF: treat as accepting the shown default
            print("", file=stream_out)
            resolved[spec.field] = None if optional else suggestion
            continue
        answer = answer.strip()
        if not answer:
            resolved[spec.field] = None if optional else suggestion
        else:
            resolved[spec.field] = _parse_threshold(answer, spec)

    print("", file=stream_out)
    return resolved


def resolve_criteria(
    args: argparse.Namespace,
    task: str,
    detection: TaskDetection,
    stream_in: TextIO,
    stream_out: TextIO,
) -> AcceptanceCriteria:
    """Combine file, flags, and prompting into the frozen criteria for this run."""
    preset: dict[str, float | None] = {}
    if args.criteria_file:
        preset.update(load_criteria_file(args.criteria_file))
    preset.update(_explicit_flag_values(args))

    applicable = {spec.field for spec in specs_for_task(task)}
    irrelevant = sorted(set(preset) - applicable)
    if irrelevant:
        print(
            f"Note: ignoring criteria that do not apply to a {task} task: "
            f"{', '.join(irrelevant)}",
            file=stream_out,
        )
    preset = {k: v for k, v in preset.items() if k in applicable}

    if args.accept_suggested:
        values = {
            spec.field: preset.get(spec.field, suggested_value(spec, task))
            for spec in specs_for_task(task)
            if spec.field in preset or spec.field not in OPTIONAL_BY_DEFAULT
        }
    elif args.non_interactive or not stream_in.isatty():
        missing = sorted(
            spec.field
            for spec in specs_for_task(task)
            if spec.field not in preset and spec.field not in OPTIONAL_BY_DEFAULT
        )
        if missing:
            raise CriteriaResolutionError(
                "Acceptance criteria are not fully specified and there is no terminal to ask on.\n"
                f"Missing: {', '.join(missing)}.\n"
                "Supply them with the matching flags, with --criteria-file, or pass "
                "--accept-suggested to use the literature-typical values as-is."
            )
        values = dict(preset)
    else:
        values = prompt_for_criteria(task, detection, preset, stream_in, stream_out)

    enabled = {k: v for k, v in values.items() if v is not None}
    if not enabled:
        raise CriteriaResolutionError(
            "Every acceptance criterion was disabled, so nothing would define success. "
            "Enable at least one."
        )
    criteria = AcceptanceCriteria(**values)

    print(RULE, file=stream_out)
    print("Acceptance criteria for this run (frozen from here on):", file=stream_out)
    for spec in specs_for_task(task):
        threshold = getattr(criteria, spec.field)
        if threshold is None:
            continue
        symbol = ">=" if spec.direction == "min" else "<="
        print(f"  {spec.label} {symbol} {threshold:g}", file=stream_out)
    print(RULE, file=stream_out)
    print("", file=stream_out)
    return criteria


# -- commands ------------------------------------------------------------------


def _peek_task(args: argparse.Namespace, stream_out: TextIO) -> TaskDetection:
    """Detect the task from the raw CSV before the full ingest, so we can ask first."""
    import pandas as pd

    frame = pd.read_csv(args.csv)
    if args.activity_col not in frame.columns:
        raise CriteriaResolutionError(
            f"Column {args.activity_col!r} not found in {args.csv}. "
            f"Available columns: {list(frame.columns)}"
        )
    forced = None if args.task == "auto" else args.task
    detection = detect_task(frame[args.activity_col], forced=forced)
    print(f"Endpoint inspection: {detection.reason}", file=stream_out)
    return detection


def command_run(
    args: argparse.Namespace, stream_in: TextIO, stream_out: TextIO
) -> int:
    load_env_file()

    dataset = Path(args.csv)
    if not dataset.exists():
        print(f"error: dataset not found: {dataset}", file=stream_out)
        return 2

    print(RULE, file=stream_out)
    print("Agentic QSAR workflow", file=stream_out)
    print(RULE, file=stream_out)
    print(f"Dataset: {dataset}", file=stream_out)

    detection = _peek_task(args, stream_out)
    criteria = resolve_criteria(args, detection.task, detection, stream_in, stream_out)

    provider = resolve_provider_name(args.llm_provider)
    try:
        client = get_llm_client(provider=args.llm_provider, model=args.llm_model)
    except LLMConfigurationError as exc:
        print(f"error: {exc}", file=stream_out)
        return 3
    print(
        f"Language model: provider={getattr(client, 'provider', provider)} "
        f"model={getattr(client, 'model', args.llm_model or resolve_model_name())}",
        file=stream_out,
    )
    if provider == "mock":
        print(
            "WARNING: the mock provider returns scripted replies. Use it for offline demos "
            "and tests only, never to evaluate real modelling decisions.",
            file=stream_out,
        )
    print("", file=stream_out)

    config = AgenticConfig(
        smiles_column=args.smiles_col,
        activity_column=args.activity_col,
        id_column=args.id_col,
        forced_task=None if args.task == "auto" else args.task,
        split_method=args.split,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        scramble_repeats=args.scramble_repeats,
        ad_k=args.ad_k,
        ad_z_factor=args.ad_z,
        random_seed=args.seed,
        max_iterations=args.max_iterations,
        budget_seconds=args.budget_seconds,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        min_compounds=args.min_compounds,
    )

    def progress(message: str) -> None:
        print(message, file=stream_out)
        stream_out.flush()

    workflow = AgenticQSARWorkflow(
        dataset_path=dataset,
        criteria=criteria,
        config=config,
        llm_client=client,
        run_id=args.run_id,
        progress=progress,
    )
    report = workflow.run()

    print("", file=stream_out)
    print(RULE, file=stream_out)
    print(
        "FINAL RESULT: "
        + ("acceptance criteria MET" if report.accepted else "acceptance criteria NOT met"),
        file=stream_out,
    )
    print(RULE, file=stream_out)
    print(report.stop_reason, file=stream_out)
    if report.best_validation is not None and report.best_verdict is not None:
        primary = report.best_validation.primary_metric
        score = report.best_validation.test_metrics.get(primary)
        print(
            f"Best iteration: {report.best_iteration_index} "
            f"(test {primary} = {'n/a' if score is None else f'{score:.4f}'})",
            file=stream_out,
        )
        for check in report.best_verdict.checks:
            print(f"  {check.render()}", file=stream_out)
    print("", file=stream_out)
    print(f"Report: {report.artifact_paths.get('report_markdown')}", file=stream_out)
    print(f"Artifacts: {report.artifact_paths.get('run_dir')}", file=stream_out)

    return 0 if report.accepted else 1


def command_show_criteria(args: argparse.Namespace, stream_out: TextIO) -> int:
    tasks = [args.task] if getattr(args, "task", None) else ["regression", "classification"]
    for task in tasks:
        print(RULE, file=stream_out)
        print(f"Suggested acceptance criteria — {task}", file=stream_out)
        print(RULE, file=stream_out)
        for spec in specs_for_task(task):
            symbol = ">=" if spec.direction == "min" else "<="
            optional = spec.field in OPTIONAL_BY_DEFAULT
            value = "off by default" if optional else f"{symbol} {suggested_value(spec, task):g}"
            print(f"  {spec.cli_flag:<28} {value}", file=stream_out)
            print(f"  {'':<28} {spec.description}", file=stream_out)
            print(f"  {'':<28} basis: {spec.source}", file=stream_out)
            print("", file=stream_out)
    print(
        "These are suggestions, not defaults. The workflow uses whatever you specify.",
        file=stream_out,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stream_out = sys.stdout
    stream_in = sys.stdin

    try:
        if args.command == "run":
            return command_run(args, stream_in, stream_out)
        if args.command == "show-criteria":
            return command_show_criteria(args, stream_out)
    except CriteriaResolutionError as exc:
        print(f"error: {exc}", file=stream_out)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=stream_out)
        return 130

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
