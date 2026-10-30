"""Run source-bound H100 PBE0 Direct/DF complete endpoints in independent processes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

SOURCE = "a3d7e5c3deff7e7590bb4cb7479b49a72a594611"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _device() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,mig.mode.current,driver_version,memory.used,power.limit,clocks.sm",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1 or "H100" not in rows[0]:
        raise RuntimeError("one assigned H100 is required")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--population-seconds", type=int, default=600)
    arguments = parser.parse_args()
    if (
        arguments.output.exists()
        or arguments.max_seconds < arguments.population_seconds
    ):
        parser.error("output must be new and deadline must admit one population")
    source = arguments.task / "source"
    library_root = arguments.task / "build-sm90-qualified"
    probe = arguments.task / "h100_sm90_endpoint_probe.py"
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if revision != SOURCE or not probe.is_file():
        raise RuntimeError("fixed source or endpoint probe is missing")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1"], text=True, timeout=30
    ).strip():
        raise RuntimeError("fixed source checkout is modified")
    driver_hash = _sha256(source / "benchmarks/readme_defaults.py")
    library_hash = _sha256(library_root / "libgenerativeqc.so")
    arguments.output.mkdir(parents=True)
    started = time.monotonic()
    device = _device()
    journal = {
        "schema": "generativeqc.h100-post2880-complete-campaign.v1",
        "status": "INCOMPLETE",
        "source_revision": SOURCE,
        "driver_sha256": driver_hash,
        "library_sha256": library_hash,
        "probe_sha256": _sha256(probe),
        "device": device,
        "max_seconds": arguments.max_seconds,
        "population_seconds": arguments.population_seconds,
        "runs": [],
    }
    journal_path = arguments.output / "campaign.json"
    _save(journal_path, journal)
    try:
        for atoms, mode, arm, repeat in itertools.product(
            (48, 96), ("direct", "df"), ("standard", "fast"), range(3)
        ):
            if (
                time.monotonic() - started + arguments.population_seconds
                > arguments.max_seconds
            ):
                raise TimeoutError("remaining deadline cannot admit a population")
            stem = f"{atoms}-{mode}-{arm}-r{repeat}"
            plan = arguments.task / "plans" / f"{stem}.json"
            parsed = json.loads(plan.read_text(encoding="utf-8"))
            if (
                parsed["source_revision"] != SOURCE
                or parsed["driver_sha256"] != driver_hash
                or parsed["native_library_sha256"] != library_hash
                or (parsed["atoms"], parsed["mode"], parsed["arm"], parsed["repeat"])
                != (atoms, mode, arm, repeat)
            ):
                raise RuntimeError(f"population plan identity mismatch: {stem}")
            environment = os.environ.copy()
            environment.update(
                PYTHONNOUSERSITE="1",
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
                SCCACHE_DIR=str(arguments.base / "cache/sccache"),
                GENERATIVEQC_LIBRARY=str(library_root / "libgenerativeqc.so"),
            )
            environment["LD_LIBRARY_PATH"] = ":".join(
                (
                    str(library_root),
                    *parsed["library_search"],
                    environment.get("LD_LIBRARY_PATH", ""),
                )
            )
            before = _device()
            destination = arguments.output / f"{stem}.json"
            command = [
                str(arguments.base / "env/bin/python"),
                str(probe),
                "--plan",
                str(plan),
                "--source-root",
                str(source),
                "--library-root",
                str(library_root),
                "--output",
                str(destination),
                "--warm-repeats",
                "3",
            ]
            started_process = time.monotonic()
            with (
                (arguments.output / f"{stem}.stdout").open("x") as stdout,
                (arguments.output / f"{stem}.stderr").open("x") as stderr,
            ):
                result = subprocess.run(
                    command,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    timeout=arguments.population_seconds,
                )
            row = {
                "stem": stem,
                "plan_sha256": _sha256(plan),
                "exit_code": result.returncode,
                "wall_seconds": time.monotonic() - started_process,
                "device_before": before,
                "device_after": _device(),
                "result_sha256": _sha256(destination) if destination.exists() else None,
            }
            journal["runs"].append(row)
            _save(journal_path, journal)
            print(json.dumps(row), flush=True)
            if result.returncode:
                raise RuntimeError(f"native complete endpoint failed: {stem}")
        journal["status"] = "PASS_SCOPED_COMPLETE_ENDPOINTS"
    except (
        OSError,
        RuntimeError,
        subprocess.TimeoutExpired,
        TimeoutError,
        ValueError,
    ) as error:
        journal.update(
            status="FAIL_OR_INCOMPLETE",
            error_type=type(error).__name__,
            error=str(error),
        )
    finally:
        journal["elapsed_seconds"] = time.monotonic() - started
        _save(journal_path, journal)
    print(json.dumps({"status": journal["status"], "runs": len(journal["runs"])}))
    if journal["status"] != "PASS_SCOPED_COMPLETE_ENDPOINTS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
