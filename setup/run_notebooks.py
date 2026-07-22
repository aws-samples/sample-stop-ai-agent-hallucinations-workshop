#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     # Notebook execution
#     "nbconvert>=7.16",
#     "nbformat>=5.10",
#     "ipykernel>=6.29",
#     # Shared demo dependencies
#     "strands-agents>=1.27.0",
#     "boto3>=1.35.0",
#     "numpy>=1.24.0",
#     "neo4j>=5.28.0",
#     "neo4j-graphrag>=1.13.0",
#     "faiss-cpu>=1.9.0",
#     "python-dotenv>=1.0.1",
#     "agent-control-sdk>=0.0.1",
#     "pyyaml>=6.0",
#     "bedrock-agentcore-starter-toolkit",
# ]
# ///
"""Execute the workshop notebooks without modifying their source files.

The default run covers demos 00 through 05. Deployment notebooks and cleanup
are opt-in because they change AWS resources.

Usage:
    uv run setup/run_notebooks.py
    uv run setup/run_notebooks.py --labs 1
    uv run setup/run_notebooks.py --labs 2-5
    uv run setup/run_notebooks.py --labs 6,7 --include-deploy
    uv run setup/run_notebooks.py --labs 8 --include-cleanup
    uv run setup/run_notebooks.py --keep-output
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "setup" / "notebook-output"
KERNEL_NAME = "hallucinations-workshop"

_INSTALL_MAGIC = re.compile(r"^(\s*)[%!]\s*(pip|pip3|conda|uv)\b")


@dataclass(frozen=True)
class Notebook:
    """A notebook included in the workshop test run."""

    lab: str
    path: Path
    deploys_resources: bool = False
    deletes_resources: bool = False


@dataclass(frozen=True)
class Result:
    """The outcome of one notebook execution."""

    notebook: Notebook
    status: str
    reason: str = ""
    detail: str = ""


NOTEBOOKS = (
    Notebook(
        "0",
        REPO_ROOT / "00-getting-started" / "getting_started_strands.ipynb",
    ),
    Notebook(
        "1",
        REPO_ROOT / "01-graphrag-demo" / "test_graphrag.ipynb",
    ),
    Notebook(
        "2",
        REPO_ROOT
        / "02-semantic-tools-demo"
        / "token_efficiency_analysis.ipynb",
    ),
    Notebook(
        "3",
        REPO_ROOT
        / "03-multiagent-demo"
        / "test_multiagent_hallucinations.ipynb",
    ),
    Notebook(
        "4",
        REPO_ROOT
        / "04-neurosymbolic-demo"
        / "test_neurosymbolic_hooks.ipynb",
    ),
    Notebook(
        "5",
        REPO_ROOT / "05-steering-demo" / "test_hooks_vs_control.ipynb",
    ),
    Notebook(
        "6",
        REPO_ROOT / "06-agentcore-boto3-demo" / "deploy_agentcore.ipynb",
        deploys_resources=True,
    ),
    Notebook(
        "7",
        REPO_ROOT
        / "07-agentcore-memory-demo"
        / "deploy_memory_agent.ipynb",
        deploys_resources=True,
    ),
    Notebook(
        "8",
        REPO_ROOT / "08-cleanup" / "cleanup.ipynb",
        deletes_resources=True,
    ),
)

KNOWN_LABS = tuple(notebook.lab for notebook in NOTEBOOKS)


def parse_labs(spec: str | None) -> set[str]:
    """Parse one lab, a comma-separated list, or a numeric range."""
    if spec is None:
        return set(KNOWN_LABS)

    selected: set[str] = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue

        if re.fullmatch(r"\d+-\d+", token):
            start, end = (int(value) for value in token.split("-"))
            if start > end:
                raise ValueError(f"invalid range '{token}': start exceeds end")
            selected.update(str(value) for value in range(start, end + 1))
        elif token.isdigit():
            selected.add(str(int(token)))
        else:
            raise ValueError(f"invalid lab '{token}'")

    unknown = selected.difference(KNOWN_LABS)
    if unknown:
        values = ", ".join(sorted(unknown, key=int))
        raise ValueError(f"unknown lab(s): {values}")
    if not selected:
        raise ValueError("no labs selected")
    return selected


def neutralize_install_magics(notebook: Any) -> int:
    """Disable package installation magics in an in-memory notebook copy."""
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        lines = []
        for line in cell.source.splitlines(keepends=True):
            match = _INSTALL_MAGIC.match(line)
            if match is None:
                lines.append(line)
                continue

            newline = "\n" if line.endswith("\n") else ""
            lines.append(
                f"{match.group(1)}# [run_notebooks] disabled: "
                f"{line.strip()}{newline}"
            )
            count += 1
        cell.source = "".join(lines)
    return count


@contextmanager
def temporary_kernel(work_dir: Path) -> Iterator[None]:
    """Expose the runner's Python environment as a temporary Jupyter kernel."""
    kernel_dir = work_dir / "kernels" / KERNEL_NAME
    kernel_dir.mkdir(parents=True)
    kernel = {
        "argv": [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": "Hallucinations Workshop Runner",
        "language": "python",
    }
    (kernel_dir / "kernel.json").write_text(
        json.dumps(kernel),
        encoding="utf-8",
    )

    previous_path = os.environ.get("JUPYTER_PATH")
    paths = [str(work_dir)]
    if previous_path:
        paths.append(previous_path)
    os.environ["JUPYTER_PATH"] = os.pathsep.join(paths)

    try:
        yield
    finally:
        if previous_path is None:
            os.environ.pop("JUPYTER_PATH", None)
        else:
            os.environ["JUPYTER_PATH"] = previous_path


def output_path(output_dir: Path, notebook: Notebook) -> Path:
    """Return an output path that preserves the source lab directory."""
    lab_dir = output_dir / notebook.path.parent.name
    lab_dir.mkdir(parents=True, exist_ok=True)
    return lab_dir / f"{notebook.path.stem}-executed.ipynb"


def cell_preview(cell: Any, limit: int = 100) -> str:
    """Return a compact description of a notebook cell."""
    first_line = next(
        (line.strip() for line in cell.source.splitlines() if line.strip()),
        "<empty cell>",
    )
    if len(first_line) <= limit:
        return first_line
    return f"{first_line[: limit - 3]}..."


def print_cell_outputs(cell: Any) -> None:
    """Print text captured by a completed notebook cell."""
    for output in cell.get("outputs", []):
        output_type = output.get("output_type")
        if output_type == "stream":
            text = output.get("text", "")
        elif output_type in {"display_data", "execute_result"}:
            text = output.get("data", {}).get("text/plain", "")
        elif output_type == "error":
            text = "\n".join(output.get("traceback", []))
        else:
            continue

        for line in str(text).rstrip().splitlines():
            print(f"    {line}", flush=True)


def run_notebook(
    notebook: Notebook,
    output_dir: Path,
    timeout: int,
) -> Result:
    """Execute one notebook and save its output away from the source file."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    relative_path = notebook.path.relative_to(REPO_ROOT)
    print(f"\nRunning {relative_path}", flush=True)

    document = None
    try:
        document = nbformat.read(notebook.path, as_version=4)
        disabled = neutralize_install_magics(document)
        if disabled:
            print(
                f"  Disabled {disabled} package-install line(s)",
                flush=True,
            )

        code_cell_indices = [
            index
            for index, cell in enumerate(document.cells)
            if cell.cell_type == "code" and cell.source.strip()
        ]
        cell_positions = {
            index: position
            for position, index in enumerate(code_cell_indices, start=1)
        }
        started_at: dict[int, float] = {}

        def on_cell_execute(*, cell: Any, cell_index: int) -> None:
            position = cell_positions[cell_index]
            started_at[cell_index] = time.monotonic()
            print(
                f"  Cell {position}/{len(code_cell_indices)} started: "
                f"{cell_preview(cell)}",
                flush=True,
            )

        def on_cell_executed(
            *,
            cell: Any,
            cell_index: int,
            execute_reply: Any,
        ) -> None:
            position = cell_positions[cell_index]
            elapsed = time.monotonic() - started_at[cell_index]
            status = execute_reply.get("content", {}).get("status", "ok")
            print(
                f"  Cell {position}/{len(code_cell_indices)} "
                f"{status} in {elapsed:.1f}s",
                flush=True,
            )
            print_cell_outputs(cell)

        executor = ExecutePreprocessor(
            timeout=timeout,
            kernel_name=KERNEL_NAME,
            allow_errors=False,
            on_cell_execute=on_cell_execute,
            on_cell_executed=on_cell_executed,
        )
        executor.preprocess(
            document,
            {"metadata": {"path": str(notebook.path.parent)}},
        )
        nbformat.write(document, output_path(output_dir, notebook))
    except Exception as exc:  # Report the failed notebook and continue the run.
        if document is not None:
            nbformat.write(document, output_path(output_dir, notebook))
        detail = traceback.format_exc()
        message = next(
            (line.strip() for line in str(exc).splitlines() if line.strip()),
            "execution failed",
        )
        reason = f"{type(exc).__name__}: {message}"
        return Result(notebook, "FAIL", reason=reason, detail=detail)

    return Result(notebook, "PASS")


def select_notebooks(
    labs: set[str],
    include_deploy: bool,
    include_cleanup: bool,
) -> list[tuple[Notebook, str | None]]:
    """Select notebooks and record why unsafe or missing ones are skipped."""
    selected = []
    for notebook in NOTEBOOKS:
        if notebook.lab not in labs:
            continue

        reason = None
        if notebook.deploys_resources and not include_deploy:
            reason = "deploys AWS resources; pass --include-deploy"
        elif notebook.deletes_resources and not include_cleanup:
            reason = "deletes AWS resources; pass --include-cleanup"
        elif not notebook.path.exists():
            reason = "notebook file not found"
        selected.append((notebook, reason))
    return selected


def print_notebooks() -> None:
    """Print the notebook registry without executing it."""
    for notebook in NOTEBOOKS:
        labels = []
        if notebook.deploys_resources:
            labels.append("deploy")
        if notebook.deletes_resources:
            labels.append("cleanup")
        suffix = f" ({', '.join(labels)})" if labels else ""
        path = notebook.path.relative_to(REPO_ROOT)
        print(f"{notebook.lab}: {path}{suffix}")


def print_summary(results: list[Result], kept_output: Path | None) -> None:
    """Print a compact result table and failure details."""
    print("\nResults")
    print("=" * 72)
    for result in results:
        path = result.notebook.path.relative_to(REPO_ROOT)
        suffix = f" ({result.reason})" if result.reason else ""
        print(f"{result.status:<4}  {path}{suffix}")

    passed = sum(result.status == "PASS" for result in results)
    failed = sum(result.status == "FAIL" for result in results)
    skipped = sum(result.status == "SKIP" for result in results)
    print(
        f"\nPassed: {passed}  Failed: {failed}  "
        f"Skipped: {skipped}  Total: {len(results)}"
    )

    for result in results:
        if result.status == "FAIL":
            path = result.notebook.path.relative_to(REPO_ROOT)
            print(f"\nFailure: {path}\n{result.detail.rstrip()}")

    if kept_output is not None:
        print(f"\nExecuted notebooks: {kept_output}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--labs",
        help="Labs to run: '4', '2,4,5', or '2-5'. Default: all.",
    )
    parser.add_argument(
        "--include-deploy",
        action="store_true",
        help="Run labs 6 and 7, which create or update AWS resources.",
    )
    parser.add_argument(
        "--include-cleanup",
        action="store_true",
        help="Run lab 8, which deletes tagged AWS resources.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-cell timeout in seconds (default: 1800).",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep executed notebooks under setup/notebook-output/.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered notebooks and exit.",
    )
    return parser


def main() -> int:
    """Run the selected notebooks and return a shell-compatible status."""
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print_notebooks()
        return 0
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    try:
        labs = parse_labs(args.labs)
    except ValueError as exc:
        parser.error(str(exc))

    plan = select_notebooks(
        labs,
        include_deploy=args.include_deploy,
        include_cleanup=args.include_cleanup,
    )

    with tempfile.TemporaryDirectory(prefix="run_notebooks_") as temp_dir:
        work_dir = Path(temp_dir)
        kept_output = None
        if args.keep_output:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            kept_output = OUTPUT_ROOT / f"{timestamp}-{os.getpid()}"
            kept_output.mkdir(parents=True)
            result_dir = kept_output
        else:
            result_dir = work_dir / "output"

        results = []
        with temporary_kernel(work_dir):
            for notebook, skip_reason in plan:
                if skip_reason is not None:
                    path = notebook.path.relative_to(REPO_ROOT)
                    print(f"\nSkipping {path}: {skip_reason}")
                    results.append(
                        Result(notebook, "SKIP", reason=skip_reason)
                    )
                    continue
                results.append(
                    run_notebook(notebook, result_dir, args.timeout)
                )

        print_summary(results, kept_output)
        return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
