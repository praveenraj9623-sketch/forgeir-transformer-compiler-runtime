"""Module entry point for the ForgeIR CPU benchmark command."""

import sys

from forgeir.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))
