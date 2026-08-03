"""CSV and static HTML rendering driven exclusively by measured benchmark JSON."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, cast

from forgeir.benchmark.types import BENCHMARK_SCHEMA_VERSION

_STATISTIC_COLUMNS = (
    "sample_count",
    "minimum_microseconds",
    "p50_microseconds",
    "p95_microseconds",
    "maximum_microseconds",
    "mean_microseconds",
    "standard_deviation_microseconds",
)


def _load_measured_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("measured benchmark JSON must contain an object")
    if document.get("benchmark_schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("measured benchmark JSON has an unsupported schema version")
    results = document.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("measured benchmark JSON must contain nonempty results")
    return cast(dict[str, Any], document)


def _summary_rows(document: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_result in cast(list[dict[str, Any]], document["results"]):
        implementation = str(raw_result["implementation"])
        end_to_end = cast(dict[str, Any], raw_result["end_to_end"])
        statistics = cast(dict[str, object], end_to_end["statistics"])
        rows.append(
            {
                "implementation": implementation,
                "scope": "end_to_end",
                "operation_id": "",
                "operation_type": "",
                "kernel": "",
                **statistics,
                "attempted_samples_per_second": raw_result["attempted_samples_per_second"],
                "planned_arena_bytes": raw_result.get("planned_arena_bytes"),
            }
        )
        for raw_operation in cast(list[dict[str, Any]], raw_result["per_operations"]):
            operation_statistics = cast(dict[str, object], raw_operation["statistics"])
            rows.append(
                {
                    "implementation": implementation,
                    "scope": str(raw_operation.get("scope", "operation")),
                    "operation_id": str(raw_operation["operation_id"]),
                    "operation_type": str(raw_operation["operation_type"]),
                    "kernel": str(raw_operation.get("kernel", "")),
                    **operation_statistics,
                    "attempted_samples_per_second": "",
                    "planned_arena_bytes": "",
                }
            )
    return rows


def _write_csv(document: dict[str, Any], path: Path) -> None:
    fields = (
        "implementation",
        "scope",
        "operation_id",
        "operation_type",
        "kernel",
        *_STATISTIC_COLUMNS,
        "attempted_samples_per_second",
        "planned_arena_bytes",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_summary_rows(document))


def _escaped(value: object) -> str:
    return html.escape(str(value), quote=True)


def _number(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return _escaped(value)


def _end_to_end_table(document: dict[str, Any]) -> str:
    rows: list[str] = []
    for result in cast(list[dict[str, Any]], document["results"]):
        statistics = cast(dict[str, object], result["end_to_end"]["statistics"])
        rows.append(
            "<tr>"
            f"<td>{_escaped(result['implementation'])}</td>"
            f"<td>{_number(statistics['p50_microseconds'])}</td>"
            f"<td>{_number(statistics['p95_microseconds'])}</td>"
            f"<td>{_number(statistics['minimum_microseconds'])}</td>"
            f"<td>{_number(statistics['maximum_microseconds'])}</td>"
            f"<td>{_number(statistics['mean_microseconds'])}</td>"
            f"<td>{_number(statistics['standard_deviation_microseconds'])}</td>"
            f"<td>{_number(result['attempted_samples_per_second'])}</td>"
            f"<td>{_number(result.get('planned_arena_bytes', 'n/a'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _operation_tables(document: dict[str, Any]) -> str:
    sections: list[str] = []
    for result in cast(list[dict[str, Any]], document["results"]):
        operation_rows: list[str] = []
        for operation in cast(list[dict[str, Any]], result["per_operations"]):
            statistics = cast(dict[str, object], operation["statistics"])
            operation_rows.append(
                "<tr>"
                f"<td>{_escaped(operation['operation_id'])}</td>"
                f"<td>{_escaped(operation['operation_type'])}</td>"
                f"<td>{_escaped(operation.get('kernel', ''))}</td>"
                f"<td>{_number(statistics['p50_microseconds'])}</td>"
                f"<td>{_number(statistics['p95_microseconds'])}</td>"
                f"<td>{_number(statistics['mean_microseconds'])}</td>"
                f"<td>{_number(statistics['standard_deviation_microseconds'])}</td>"
                "</tr>"
            )
        coverage = _escaped(result.get("operation_profile_contract", "not available"))
        rows = "\n".join(operation_rows)
        sections.append(
            f"<h3>{_escaped(result['implementation'])}</h3>"
            f'<p class="note">{coverage}</p>'
            "<table><thead><tr><th>ID</th><th>Type</th><th>Kernel</th>"
            "<th>p50 (us)</th><th>p95 (us)</th><th>Mean (us)</th>"
            f"<th>Stddev (us)</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    return "\n".join(sections)


def _environment_table(document: dict[str, Any]) -> str:
    environment = cast(dict[str, Any], document["environment"])
    cpu = cast(dict[str, object], environment["cpu"])
    values = (
        ("CPU", cpu["model"]),
        ("Physical cores", cpu["physical_core_count"]),
        ("Logical cores", cpu["logical_core_count"]),
        ("Operating system", environment["operating_system"]["platform"]),
        ("Compiler", environment["compiler"]),
        ("Build type", environment["build_type"]),
        ("PyTorch", environment["pytorch_version"]),
        ("Python", environment["python_version"]),
        ("Git commit", environment["git"]["commit"]),
        ("Git worktree dirty", environment["git"]["worktree_dirty"]),
    )
    return "\n".join(
        f"<tr><th>{_escaped(name)}</th><td>{_escaped(value)}</td></tr>" for name, value in values
    )


def _write_html(document: dict[str, Any], path: Path) -> None:
    conclusions = "\n".join(
        f"<li>{_escaped(item)}</li>" for item in cast(list[str], document["conclusions"])
    )
    protocol = cast(dict[str, Any], document["protocol"])
    content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ForgeIR CPU benchmark report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #17202a; }}
h1, h2, h3 {{ color: #1b4f72; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #ccd1d1; padding: .45rem .6rem; text-align: right; }}
th:first-child, td:first-child, td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
thead {{ background: #eaf2f8; }}
.note {{ color: #566573; }}
.warning {{ background: #fef9e7; border-left: .3rem solid #f1c40f; padding: .8rem; }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body>
<h1>ForgeIR CPU benchmark report</h1>
<p>Measured at {_escaped(document["measured_at_utc"])}. Report schema
{_escaped(document["benchmark_schema_version"])}.</p>
<p class="warning">These are local measurements from the captured host. They are not universal
performance claims. Setup, graph parsing, optimization, session loading, correctness checks, and
report rendering are excluded from end-to-end latency.</p>
<h2>Measured conclusions</h2>
<ul>{conclusions}</ul>
<h2>Protocol</h2>
<p>Warm-ups: {_escaped(protocol["warmup_count"])}; measured iterations:
{_escaped(protocol["measured_iteration_count"])}; batch size:
{_escaped(protocol["input_shapes"]["block_input"][0])}.</p>
<p>ForgeIR clock: {_escaped(protocol["clock_sources"]["forgeir"])}. PyTorch end-to-end clock:
{_escaped(protocol["clock_sources"]["pytorch_end_to_end"])}.</p>
<h2>Environment</h2>
<table><tbody>{_environment_table(document)}</tbody></table>
<h2>End-to-end measured execution</h2>
<table><thead><tr><th>Implementation</th><th>p50 (us)</th><th>p95 (us)</th>
<th>Min (us)</th><th>Max (us)</th><th>Mean (us)</th><th>Stddev (us)</th>
<th>Attempted samples/s</th><th>Planned arena bytes</th></tr></thead>
<tbody>{_end_to_end_table(document)}</tbody></table>
<h2>Operation-level profiles</h2>
{_operation_tables(document)}
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def render_reports_from_json(measured_json: Path) -> tuple[Path, Path]:
    """Generate CSV and HTML by reading only an already-written measured JSON report."""
    document = _load_measured_json(measured_json)
    csv_path = measured_json.with_name("benchmark_summary.csv")
    html_path = measured_json.with_name("benchmark_report.html")
    _write_csv(document, csv_path)
    _write_html(document, html_path)
    return csv_path, html_path
