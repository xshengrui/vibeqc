"""Retain separate complete GPU4PySCF PBE0 populations on one H100."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import statistics
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
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--population-seconds", type=int, default=600)
    arguments = parser.parse_args()
    if (
        arguments.output.exists()
        or arguments.max_seconds < arguments.population_seconds
    ):
        parser.error("output must be new and deadline must admit one population")
    source = arguments.task / "source"
    probe = arguments.task / "h100_sm90_gpu4_control.py"
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if revision != SOURCE or not probe.is_file():
        raise RuntimeError("fixed source or GPU4PySCF control is missing")
    arguments.output.mkdir(parents=True)
    started = time.monotonic()
    device = _device()
    journal = {
        "schema": "generativeqc.h100-post2880-gpu4-campaign.v1",
        "status": "INCOMPLETE",
        "source_revision": SOURCE,
        "probe_sha256": _sha256(probe),
        "driver_sha256": _sha256(source / "benchmarks/readme_defaults.py"),
        "device": device,
        "runs": [],
    }
    journal_path = arguments.output / "campaign.json"
    _save(journal_path, journal)
    try:
        for atoms, mode, repeat in itertools.product(
            (48, 96), ("direct", "df"), range(3)
        ):
            if (
                time.monotonic() - started + arguments.population_seconds
                > arguments.max_seconds
            ):
                raise TimeoutError("campaign deadline cannot admit another population")
            stem = f"{atoms}-{mode}-r{repeat}"
            plan = arguments.task / "plans" / f"{atoms}-{mode}-gpu4pyscf-r{repeat}.json"
            parsed = json.loads(plan.read_text(encoding="utf-8"))
            if (
                parsed["source_revision"] != SOURCE
                or parsed["driver_sha256"] != journal["driver_sha256"]
                or (parsed["atoms"], parsed["mode"], parsed["arm"], parsed["repeat"])
                != (atoms, mode, "gpu4pyscf", repeat)
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
            )
            environment["LD_LIBRARY_PATH"] = ":".join(
                (*parsed["library_search"], environment.get("LD_LIBRARY_PATH", ""))
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
                "--output",
                str(destination),
            ]
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
            after = _device()
            row = {
                "stem": stem,
                "plan_sha256": _sha256(plan),
                "exit_code": result.returncode,
                "device_before": before,
                "device_after": after,
                "result_sha256": _sha256(destination) if destination.exists() else None,
            }
            journal["runs"].append(row)
            _save(journal_path, journal)
            print(json.dumps(row), flush=True)
            if result.returncode:
                raise RuntimeError(f"GPU4PySCF population failed: {stem}")
        journal["status"] = "PASS_SCOPED_REFERENCE_ENDPOINTS"
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
    if journal["status"] != "PASS_SCOPED_REFERENCE_ENDPOINTS":
        raise SystemExit(2)
    summary = []
    for atoms, mode in itertools.product((48, 96), ("direct", "df")):
        results = [
            json.loads(
                (arguments.output / f"{atoms}-{mode}-r{repeat}.json").read_text()
            )
            for repeat in range(3)
        ]
        cold = [item["samples"][0]["seconds"] for item in results]
        warm = [
            statistics.median(sample["seconds"] for sample in item["samples"][1:])
            for item in results
        ]
        summary.append(
            {
                "atoms": atoms,
                "mode": mode,
                "cold_seconds": cold,
                "warm_process_medians_seconds": warm,
                "cold_median": statistics.median(cold),
                "warm_median": statistics.median(warm),
                "cold_population_sd": statistics.pstdev(cold),
                "warm_population_sd": statistics.pstdev(warm),
            }
        )
    _save(
        arguments.output / "summary.json",
        {"source_revision": SOURCE, "groups": summary},
    )
    print(json.dumps({"status": journal["status"], "groups": len(summary)}))


if __name__ == "__main__":
    main()
