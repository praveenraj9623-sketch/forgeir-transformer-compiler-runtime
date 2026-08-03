"""Python access to ForgeIR diagnostics and read-only graph inspection."""

from forgeir_py import doctor, graph_summary, version

__version__ = version()

__all__ = ["__version__", "doctor", "graph_summary", "version"]
