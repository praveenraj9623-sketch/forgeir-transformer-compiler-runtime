"""Python access to the ForgeIR Milestone 1 diagnostics binding."""

from forgeir_py import doctor, version

__version__ = version()

__all__ = ["__version__", "doctor", "version"]
