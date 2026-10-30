"""Run finite, interleaved H100 PBE0 baseline/candidate populations.

Each child process owns one frozen #2863 population plan. The pair uses one
assigned GPU and identical input, oracle, core library, image and CUDA context
rules. Results are journaled after every child; a failure stops the campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import time
from pathlib import Path


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _device_snapshot() -> dict:
    query = (
        "name,uuid,compute_cap,mig.mode.current,driver_version,memory.total,"
        "memory.used,power.limit,power.draw,clocks.sm,clocks.mem,utilization.gpu"
    )
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    rows = completed.stdout.strip().splitlines()
    if not rows or not all("H100" in row for row in rows):
        raise RuntimeError("campaign requires H100 GPU visibility")
    return {"query": query, "rows": rows}


def _plan(task: Path, atoms: int, mode: str, arm: str, repeat: int) -> Path:
    plan = task / "plans" / f"{atoms}-{mode}-{arm}-r{repeat}.json"
    if not plan.is_file():
        raise RuntimeError(f"missing post-#2880 plan: {plan}")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=13_800)
    parser.add_argument("--population-seconds", type=int, default=900)
    arguments = parser.parse_args()
    if (
        arguments.max_seconds < arguments.population_seconds
        or arguments.output.exists()
    ):
        parser.error("output must be new and deadline must admit one population")
    arguments.output.mkdir(parents=True)
    probe = arguments.probe or arguments.task / "h100_pbe0_phase_probe.py"
    control = arguments.task / "schedule-probe.py"
    source = arguments.task / "source"
    library_root = arguments.task / "build-sm90-qualified"
    if not all(path.exists() for path in (probe, control, source, library_root)):
        raise RuntimeError("fixed-source qualification assets are incomplete")
    initial_device = _device_snapshot()
    started = time.monotonic()
    record = {
        "schema": "generativeqc.h100-pbe0-interleaved.v1",
        "status": "INCOMPLETE",
        "base": str(arguments.base),
        "task": str(arguments.task),
        "probe_sha256": _hash(probe),
        "control_sha256": _hash(control),
        "source_revision": "a3d7e5c3deff7e7590bb4cb7479b49a72a594611",
        "library_sha256": _hash(library_root / "libgenerativeqc.so"),
        "candidate_planner_sha256": _hash(
            source / "python/generativeqc_compiler/method/stationary_resources.py"
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": initial_device,
        "max_seconds": arguments.max_seconds,
        "population_seconds": arguments.population_seconds,
        "runs": [],
    }
    journal = arguments.output / "campaign.json"
    _write(journal, record)
    try:
        for atoms, mode, arm in itertools.product(
            (48, 96), ("direct", "df"), ("standard", "fast")
        ):
            for repeat in range(3):
                plan = _plan(arguments.task, atoms, mode, arm, repeat)
                parsed = json.loads(plan.read_text(encoding="utf-8"))
                if parsed[
                    "source_revision"
                ] != "a3d7e5c3deff7e7590bb4cb7479b49a72a594611" or (
                    parsed["atoms"],
                    parsed["mode"],
                    parsed["arm"],
                ) != (atoms, mode, arm):
                    raise RuntimeError("post-#2880 population plan identity mismatch")
                if parsed["native_library_sha256"] != record["library_sha256"]:
                    raise RuntimeError("plan native core differs from fixed build")
                order = (
                    ("baseline", "candidate")
                    if repeat % 2 == 0
                    else (
                        "candidate",
                        "baseline",
                    )
                )
                for variant in order:
                    if time.monotonic() - started + arguments.population_seconds > (
                        arguments.max_seconds
                    ):
                        raise TimeoutError(
                            "remaining campaign deadline cannot admit a population"
                        )
                    stem = f"{atoms}-{mode}-{arm}-r{repeat}-{variant}"
                    destination = arguments.output / f"{stem}.json"
                    if destination.exists():
                        raise FileExistsError(destination)
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
                    command = [
                        str(arguments.base / "env/bin/python"),
                        str(control),
                        "--plan",
                        str(plan),
                        "--probe",
                        str(probe),
                        "--source-root",
                        str(source),
                        "--library-root",
                        str(library_root),
                        "--selection",
                        "automatic" if variant == "candidate" else "serial",
                        "--output",
                        str(destination),
                        "--repeats",
                        "3",
                    ]
                    device_before = _device_snapshot()
                    begin = time.monotonic()
                    with (
                        (arguments.output / f"{stem}.stdout").open("x") as stdout,
                        (arguments.output / f"{stem}.stderr").open("x") as stderr,
                    ):
                        completed = subprocess.run(
                            command,
                            env=environment,
                            stdout=stdout,
                            stderr=stderr,
                            timeout=arguments.population_seconds,
                            check=False,
                        )
                    row = {
                        "stem": stem,
                        "variant": variant,
                        "plan_sha256": _hash(plan),
                        "exit_code": completed.returncode,
                        "wall_seconds": time.monotonic() - begin,
                        "device_before": device_before,
                        "device_after": _device_snapshot(),
                        "result_sha256": _hash(destination)
                        if destination.exists()
                        else None,
                    }
                    record["runs"].append(row)
                    _write(journal, record)
                    print(json.dumps(row), flush=True)
                    if completed.returncode:
                        raise RuntimeError(f"population failed: {stem}")
        record["status"] = "PASS_SCOPED_NATIVE_PAIRS"
    except (
        OSError,
        RuntimeError,
        subprocess.TimeoutExpired,
        TimeoutError,
        ValueError,
    ) as error:
        record["status"] = "FAIL_OR_INCOMPLETE"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        _write(journal, record)
    print(json.dumps({"status": record["status"], "runs": len(record["runs"])}))
    if record["status"] != "PASS_SCOPED_NATIVE_PAIRS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
