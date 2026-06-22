"""Regenerate every figure used by the quantum-drift submission."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _code_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run([sys.executable, "make_paper_figures.py"], cwd=_code_dir(),
                   env=env, check=True)


if __name__ == "__main__":
    main()
