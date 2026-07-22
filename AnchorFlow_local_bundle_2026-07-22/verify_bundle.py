#!/usr/bin/env python3
"""Verify the downloaded AnchorFlow bundle and its portable task paths."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKSUMS = ROOT / "MANIFEST.sha256"
TASK_MANIFEST = (
    ROOT
    / "repro_runs"
    / "2026-06-22-full-repro"
    / "task_manifest_mainline_a.csv"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums() -> int:
    checked = 0
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"Missing bundle file: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise AssertionError(f"Checksum mismatch: {relative}")
        checked += 1
    return checked


def verify_task_paths() -> int:
    text = TASK_MANIFEST.read_text(encoding="utf-8")
    if "/root/code/CONCERT/" in text:
        raise AssertionError("Portable manifest still contains the server path")

    checked = 0
    with TASK_MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for field in ("source_data_file", "target_data_file", "selected_cell_file"):
                path = ROOT / row[field]
                if not path.is_file():
                    raise AssertionError(
                        f"Missing input for {row['task_id']}: {row[field]}"
                    )
                checked += 1
    return checked


def main() -> None:
    file_count = verify_checksums()
    task_path_count = verify_task_paths()
    print(
        f"OK: {file_count} files matched SHA-256; "
        f"{task_path_count} task input references resolved"
    )


if __name__ == "__main__":
    main()
