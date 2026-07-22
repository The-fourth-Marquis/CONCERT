#!/usr/bin/env python3
"""Dependency-free test runner for the isolated prototype."""

from pathlib import Path
import subprocess
import sys


TEST_ROOT = Path(__file__).resolve().parent


def main() -> None:
    tests = sorted(TEST_ROOT.glob("test_*.py"))
    if not tests:
        raise SystemExit("no tests discovered")
    for test in tests:
        print(f"RUN {test.name}", flush=True)
        subprocess.run([sys.executable, str(test)], check=True)
    print(f"PASS {len(tests)} test files", flush=True)


if __name__ == "__main__":
    main()
