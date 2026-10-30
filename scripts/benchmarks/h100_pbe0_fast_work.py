"""Retain actual H100 PBE0 Fast precision work using the frozen endpoint driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("output must not exist")
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    if plan["arm"] != "fast" or plan["mode"] not in ("direct", "df"):
        parser.error("a native Fast Direct/DF plan is required")
    library = arguments.library_root / "libgenerativeqc.so"
    driver = arguments.source_root / "benchmarks/readme_defaults.py"
    if (
        _sha256(driver) != plan["driver_sha256"]
        or _sha256(library) != plan["native_library_sha256"]
    ):
        raise RuntimeError("source or library differs from plan")
    if Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve() != library.resolve():
        raise RuntimeError("loaded native library differs from plan")
    sys.path[:0] = [str(arguments.source_root), str(arguments.source_root / "python")]
    import cupy as cp

    from benchmarks import readme_defaults as frozen
    from benchmarks.readme_defaults_accuracy import errors
    from benchmarks.readme_hf_scaling import scaling_cases

    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError("one assigned GPU required")
    properties = cp.cuda.runtime.getDeviceProperties(0)
    if (properties["major"], properties["minor"]) != (9, 0):
        raise RuntimeError("SM90 device required")
    atoms = scaling_cases()[f"water-{plan['atoms']}"].atoms
    if json.loads(json.dumps(atoms)) != plan["geometry_bohr"]:
        raise RuntimeError("molecular input differs from plan")
    record = {
        "schema": "generativeqc.h100-pbe0-fast-work.v1",
        "status": "INCOMPLETE",
        "source_revision": plan["source_revision"],
        "plan_sha256": _sha256(arguments.plan),
        "probe_sha256": _sha256(Path(__file__)),
        "driver_sha256": _sha256(driver),
        "library_sha256": _sha256(library),
        "arm": "fast",
        "mode": plan["mode"],
        "atoms": plan["atoms"],
        "samples": [],
    }
    try:
        options = SimpleNamespace(
            method="pbe0",
            arm="fast",
            mode=plan["mode"],
            inputs=Path(plan["task_root"]) / "data/pbe0-frozen-inputs",
            warm_repeats=3,
        )
        frozen.run_native(options, atoms, record, record["samples"].append)
        if len(record["samples"]) != 4:
            raise RuntimeError("incomplete Fast cold/warm endpoints")
        for sample in record["samples"]:
            comparison = errors(
                sample["energy"],
                sample["forces"],
                plan["gpu_l5_reference"],
                plan["atoms"],
            )
            sample["independent_oracle"] = comparison
            if not comparison["passed"]:
                raise RuntimeError("Fast independent energy/force gate failed")
        record["status"] = "PASS_SCOPED_FAST_WORK"
    except (
        ArithmeticError,
        AttributeError,
        ImportError,
        MemoryError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        record.update(
            status="FAIL",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    finally:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": record["status"], "output": str(arguments.output)}))
    if record["status"] != "PASS_SCOPED_FAST_WORK":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
