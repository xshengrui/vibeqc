"""Run a frozen PBE0 endpoint probe with one qualification-only SM90 schedule control."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

SOURCE = "a3d7e5c3deff7e7590bb4cb7479b49a72a594611"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(source: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection",
        choices=("automatic", "cooperative_no_phase", "serial"),
        required=True,
    )
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()

    if arguments.output.exists() or arguments.repeats < 1:
        parser.error("output must be new and repeats must be positive")
    if _git(arguments.source_root, "rev-parse", "HEAD") != SOURCE:
        raise RuntimeError("source does not match the fixed post-#2880 baseline")
    if _git(arguments.source_root, "status", "--porcelain=v1"):
        raise RuntimeError("fixed source is modified")
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    if plan["source_revision"] != SOURCE:
        raise RuntimeError("population plan has a different source revision")
    if (
        _sha256(arguments.source_root / "benchmarks/readme_defaults.py")
        != plan["driver_sha256"]
    ):
        raise RuntimeError("benchmark driver differs from the population plan")
    if (
        _sha256(arguments.library_root / "libgenerativeqc.so")
        != plan["native_library_sha256"]
    ):
        raise RuntimeError("native library differs from the population plan")

    sys.path[:0] = [str(arguments.source_root), str(arguments.source_root / "python")]
    from generativeqc import _stationary_cuda

    original = _stationary_cuda.plan_stationary_cuda_resources
    calls: Counter[tuple[int, int, int]] = Counter()

    def select_resources(*call_args: object, **call_kwargs: object) -> object:
        target = call_kwargs["target"]
        if target.compute_capability != (9, 0):
            raise RuntimeError("schedule control is restricted to SM90")
        if call_kwargs.get("cooperative_becke") is not None:
            raise RuntimeError("an explicit production selection bypassed the control")
        if arguments.selection == "serial":
            call_kwargs["cooperative_becke"] = False
            call_kwargs["phased_becke"] = False
        elif arguments.selection == "cooperative_no_phase":
            call_kwargs["phased_becke"] = False
        resources = original(*call_args, **call_kwargs)
        calls[
            (
                int(call_kwargs["atoms"]),
                resources.becke_threads_per_point,
                resources.phased_becke_bytes,
            )
        ] += 1
        return resources

    probe_args = [
        str(arguments.probe),
        "--plan",
        str(arguments.plan),
        "--source-root",
        str(arguments.source_root),
        "--library-root",
        str(arguments.library_root),
        "--output",
        str(arguments.output),
        "--repeats",
        str(arguments.repeats),
    ]
    previous_args = sys.argv
    try:
        sys.argv = probe_args
        with patch.object(
            _stationary_cuda, "plan_stationary_cuda_resources", select_resources
        ):
            runpy.run_path(str(arguments.probe), run_name="__main__")
    finally:
        sys.argv = previous_args

    if arguments.output.exists():
        result = json.loads(arguments.output.read_text(encoding="utf-8"))
        result["schedule_control"] = {
            "selection": arguments.selection,
            "source_revision": SOURCE,
            "control_sha256": _sha256(Path(__file__)),
            "probe_sha256": _sha256(arguments.probe),
            "planner_calls": [
                {
                    "atoms": atoms,
                    "threads_per_point": threads,
                    "phased_bytes": phase_bytes,
                    "count": count,
                }
                for (atoms, threads, phase_bytes), count in sorted(calls.items())
            ],
        }
        arguments.output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
