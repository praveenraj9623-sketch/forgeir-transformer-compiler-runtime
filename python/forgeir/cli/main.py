"""ForgeIR command-line entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from forgeir.benchmark import BenchmarkArtifacts, run_cpu_benchmark
from forgeir.pipeline import PipelineResult, StageStatus, run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgeir",
        description="ForgeIR deterministic compiler and CPU-runtime workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pipeline = subparsers.add_parser(
        "pipeline",
        help="Run generate, export, verification, optimization, planning, execution, and parity.",
        description=(
            "Run the complete fail-fast ForgeIR workflow in an isolated config-hash-derived "
            "directory. Existing runs require explicit --force replacement."
        ),
    )
    pipeline.add_argument("--config", type=Path, required=True, help="Pipeline JSON config path.")
    pipeline.add_argument(
        "--output-dir", type=Path, required=True, help="Root directory for the isolated run."
    )
    pipeline.add_argument(
        "--force",
        action="store_true",
        help="Replace the exact config-derived run directory if it already exists.",
    )
    pipeline.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned stages without creating or executing artifacts.",
    )
    presentation = pipeline.add_mutually_exclusive_group()
    presentation.add_argument(
        "--json", action="store_true", help="Write the complete machine-readable result as JSON."
    )
    presentation.add_argument(
        "--quiet", action="store_true", help="Suppress all normal and diagnostic console output."
    )
    pipeline.add_argument(
        "--verbose",
        action="store_true",
        help="Include artifact hashes and complete diagnostics in human-readable output.",
    )
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Measure PyTorch eager CPU and ForgeIR CPU O0/O1/O2 reproducibly.",
        description=(
            "Run a setup-separated CPU benchmark, retain raw samples, and render JSON-derived "
            "CSV and static HTML reports. Existing output directories require --force."
        ),
    )
    benchmark.add_argument("--config", type=Path, required=True, help="Benchmark JSON config path.")
    benchmark.add_argument(
        "--output-dir", type=Path, required=True, help="New benchmark report directory."
    )
    benchmark.add_argument(
        "--force", action="store_true", help="Replace the exact requested output directory."
    )
    benchmark_presentation = benchmark.add_mutually_exclusive_group()
    benchmark_presentation.add_argument(
        "--json", action="store_true", help="Write result paths and conclusions as JSON."
    )
    benchmark_presentation.add_argument(
        "--quiet", action="store_true", help="Suppress normal benchmark console output."
    )
    return parser


def _render_human(result: PipelineResult, *, verbose: bool) -> None:
    console = Console(stderr=not result.success)
    title = "ForgeIR pipeline dry run" if result.dry_run else "ForgeIR pipeline"
    outcome = "planned" if result.dry_run else ("success" if result.success else "failed")
    outcome_style = "cyan" if result.dry_run else ("green" if result.success else "red")
    console.print(f"[bold]{title}[/bold]: [{outcome_style}]{outcome}[/{outcome_style}]")
    console.print(f"Run ID: [bold]{result.run_id}[/bold]")
    console.print(f"Run directory: {result.run_directory}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Artifacts", justify="right")
    for index, stage in enumerate(result.stages, 1):
        style = {
            StageStatus.SUCCEEDED: "green",
            StageStatus.FAILED: "red",
            StageStatus.SKIPPED: "dim",
            StageStatus.PLANNED: "cyan",
        }[stage.status]
        table.add_row(
            str(index),
            stage.stage.value,
            f"[{style}]{stage.status.value}[/{style}]",
            str(len(stage.artifacts)),
        )
    console.print(table)

    if result.manifest_path is not None:
        console.print(f"Manifest: {result.manifest_path}")
    if result.status_path is not None:
        console.print(f"Status: {result.status_path}")
    if result.diagnostics:
        console.print("[bold red]Diagnostics[/bold red]")
        for diagnostic in result.diagnostics:
            console.print_json(json.dumps(diagnostic, sort_keys=True))
    if verbose:
        for stage in result.stages:
            if not stage.artifacts and not stage.diagnostics:
                continue
            console.print(f"[bold]{stage.stage.value}[/bold]")
            for artifact in stage.artifacts:
                console.print(
                    f"  {artifact.path}  sha256={artifact.sha256}  bytes={artifact.size_bytes}"
                )
            for diagnostic in stage.diagnostics:
                console.print_json(json.dumps(diagnostic, sort_keys=True))


def _benchmark_document(result: BenchmarkArtifacts) -> dict[str, object]:
    return {
        "success": result.success,
        "output_directory": str(result.output_directory),
        "measured_json": str(result.measured_json),
        "summary_csv": str(result.summary_csv),
        "report_html": str(result.report_html),
        "manifest": str(result.manifest),
        "conclusions": list(result.conclusions),
    }


def _render_benchmark(result: BenchmarkArtifacts) -> None:
    console = Console(stderr=not result.success)
    style = "green" if result.success else "red"
    status = "success" if result.success else "failed correctness or gross-failure threshold"
    console.print(f"[bold]ForgeIR CPU benchmark[/bold]: [{style}]{status}[/{style}]")
    for conclusion in result.conclusions:
        console.print(f"- {conclusion}")
    table = Table(show_header=False)
    table.add_column("Artifact", style="bold")
    table.add_column("Path")
    table.add_row("Measured JSON", str(result.measured_json))
    table.add_row("Summary CSV", str(result.summary_csv))
    table.add_row("Static HTML", str(result.report_html))
    table.add_row("Manifest", str(result.manifest))
    console.print(table)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the source or installed ForgeIR CLI."""
    parser = _parser()
    args = parser.parse_args(arguments)
    if args.command == "benchmark":
        try:
            benchmark = run_cpu_benchmark(args.config, args.output_dir, force=args.force)
        except Exception as error:
            failure = {
                "success": False,
                "diagnostics": [
                    {
                        "code": "benchmark_failed",
                        "message": str(error),
                        "details": {"exception_type": type(error).__name__},
                    }
                ],
            }
            if not args.quiet:
                if args.json:
                    print(json.dumps(failure, indent=2, sort_keys=True))
                else:
                    console = Console(stderr=True)
                    console.print("[bold red]ForgeIR CPU benchmark failed[/bold red]")
                    console.print_json(json.dumps(failure, sort_keys=True))
            return 2
        if not args.quiet:
            if args.json:
                print(json.dumps(_benchmark_document(benchmark), indent=2, sort_keys=True))
            else:
                _render_benchmark(benchmark)
        return 0 if benchmark.success else 1
    if args.command != "pipeline":
        parser.error(f"unsupported command {args.command!r}")
    try:
        result = run_pipeline(
            args.config,
            args.output_dir,
            force=args.force,
            dry_run=args.dry_run,
        )
    except Exception as error:
        failure = {
            "pipeline_schema_version": "1.0",
            "success": False,
            "failure_stage": None,
            "diagnostics": [
                {
                    "code": "pipeline_setup_failed",
                    "message": str(error),
                    "details": {"exception_type": type(error).__name__},
                }
            ],
        }
        if not args.quiet:
            if args.json:
                print(json.dumps(failure, indent=2, sort_keys=True))
            else:
                console = Console(stderr=True)
                console.print("[bold red]ForgeIR pipeline setup failed[/bold red]")
                console.print_json(json.dumps(failure, sort_keys=True))
        return 2

    if not args.quiet:
        if args.json:
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        else:
            _render_human(result, verbose=args.verbose)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
