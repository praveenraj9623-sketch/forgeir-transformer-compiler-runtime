"""Python access to ForgeIR compilation, planning, and CPU execution."""

from forgeir.benchmark import BenchmarkArtifacts, BenchmarkConfig, run_cpu_benchmark
from forgeir.pipeline import PipelineConfig, PipelineResult, PipelineStage, run_pipeline
from forgeir_py import doctor, graph_summary, verify_graph, version

__version__ = version()

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    "BenchmarkArtifacts",
    "BenchmarkConfig",
    "__version__",
    "doctor",
    "graph_summary",
    "run_cpu_benchmark",
    "run_pipeline",
    "verify_graph",
    "version",
]
