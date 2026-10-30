"""Measure one fixed-source H100 PBE0 complete E+F population without phase probes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

SOURCE = "a3d7e5c3deff7e7590bb4cb7479b49a72a594611"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _errors(energy: float, forces: Any, reference: dict, atoms: int) -> dict:
    differences = [
        float(actual) - expected
        for row, expected_row in zip(forces, reference["forces"], strict=True)
        for actual, expected in zip(row, expected_row, strict=True)
    ]
    if len(differences) != atoms * 3:
        raise RuntimeError("independent force vector dimensions differ")
    return {
        "energy_abs_per_atom_hartree": abs(energy - reference["energy"]) / atoms,
        "force_max_abs_hartree_per_bohr": max(map(abs, differences)),
        "force_rms_hartree_per_bohr": math.sqrt(
            math.fsum(value * value for value in differences) / len(differences)
        ),
    }


class _OracleImportBlock(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split(".")[0] in {"pyscf", "gpu4pyscf"}:
            raise RuntimeError(f"native endpoint attempted oracle import: {fullname}")


def _force_route(owner: Any, arm: str) -> dict:
    if arm == "fast":
        return {
            "native_prepared_force_eligible": None,
            "generated_force_count": None,
            "status": "UNAVAILABLE_FAST_OWNER_BOUNDARY",
        }
    supported = owner._native_dft_force_eligible(0)
    diagnostics = owner.resource_diagnostics
    generated = None if diagnostics is None else diagnostics.get("generated_force")
    return {
        "native_prepared_force_eligible": supported,
        "generated_force_count": len(generated)
        if isinstance(generated, list)
        else None,
        "status": "MEASURED_POST_ENDPOINT",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-repeats", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.warm_repeats < 1:
        parser.error("output must be new and warm repeats must be positive")
    revision = subprocess.check_output(
        ["git", "-C", str(arguments.source_root), "rev-parse", "HEAD"],
        text=True,
        timeout=30,
    ).strip()
    if revision != SOURCE:
        raise RuntimeError("source differs from the fixed post-#2880 baseline")
    if subprocess.check_output(
        ["git", "-C", str(arguments.source_root), "status", "--porcelain=v1"],
        text=True,
        timeout=30,
    ).strip():
        raise RuntimeError("fixed source checkout is modified")
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    if plan["source_revision"] != SOURCE or plan["arm"] not in ("standard", "fast"):
        raise RuntimeError("plan is not a fixed-source native PBE0 population")
    driver = arguments.source_root / "benchmarks/readme_defaults.py"
    library = arguments.library_root / "libgenerativeqc.so"
    if (
        _sha256(driver) != plan["driver_sha256"]
        or _sha256(library) != plan["native_library_sha256"]
    ):
        raise RuntimeError("driver or native library differs from plan")
    if Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve() != library.resolve():
        raise RuntimeError("runtime native library differs from plan")

    sys.path[:0] = [str(arguments.source_root), str(arguments.source_root / "python")]
    sys.meta_path.insert(0, _OracleImportBlock())
    import cupy as cp
    import numpy as np

    from benchmarks import readme_defaults as defaults
    from benchmarks.readme_hf_scaling import scaling_cases

    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode()
    if cp.cuda.runtime.getDeviceCount() != 1 or (
        properties["major"],
        properties["minor"],
    ) != (9, 0):
        raise RuntimeError("one assigned SM90 device is required")
    atoms = scaling_cases()[f"water-{plan['atoms']}"].atoms
    if json.loads(json.dumps(atoms)) != plan["geometry_bohr"]:
        raise RuntimeError("molecular geometry differs from plan")
    reference = plan["gpu_l5_reference"]
    gates = plan["gates"]
    stability = _errors(
        plan["gpu_l4_reference"]["energy"],
        plan["gpu_l4_reference"]["forces"],
        reference,
        plan["atoms"],
    )
    if any(stability[key] > limit / 4 for key, limit in gates.items()):
        raise RuntimeError("independent oracle grid stability gate failed")

    record = {
        "schema": "generativeqc.h100-post2880-complete-endpoints.v1",
        "status": "INCOMPLETE",
        "source_revision": SOURCE,
        "plan_sha256": _sha256(arguments.plan),
        "probe_sha256": _sha256(Path(__file__)),
        "driver_sha256": _sha256(driver),
        "library_sha256": _sha256(library),
        "aot_sha256": {
            filename: _sha256(arguments.library_root / filename)
            for filename in (
                "generativeqc_stationary_pbe0_rks.json",
                "generativeqc_stationary_pbe0_rks_spd.json",
                "libgenerativeqc_stationary_pbe0_rks.so",
                "libgenerativeqc_stationary_pbe0_rks_spd.so",
            )
        },
        "device": {
            "name": name,
            "runtime": cp.cuda.runtime.runtimeGetVersion(),
            "driver": cp.cuda.runtime.driverGetVersion(),
        },
        "atoms": plan["atoms"],
        "mode": plan["mode"],
        "arm": plan["arm"],
        "oracle_stability": stability,
        "clean_ef": [],
    }
    _save(arguments.output, record)
    owner = None
    try:
        cp.cuda.runtime.deviceSynchronize()
        started = perf_counter()
        calculator = defaults.native_calculator(
            "pbe0",
            plan["arm"],
            plan["mode"],
            Path(plan["task_root"]) / "data/pbe0-frozen-inputs",
        )
        owner = calculator.prepare_batch([atoms], warm_start=True)
        for repeat in range(arguments.warm_repeats + 1):
            if repeat:
                cp.cuda.runtime.deviceSynchronize()
                started = perf_counter()
            item = owner.execute(strict=True, properties=("energy", "forces")).items[0]
            cp.cuda.runtime.deviceSynchronize()
            seconds = perf_counter() - started
            if not item.converged or item.executed_backend != "cuda":
                raise RuntimeError("native CUDA endpoint did not converge")
            energy = float(item.energy)
            forces = np.asarray(item.forces).tolist()
            errors = _errors(energy, forces, reference, plan["atoms"])
            sample = {
                "phase": "cold" if repeat == 0 else "warm",
                "repeat": max(0, repeat - 1),
                "seconds": seconds,
                "energy": energy,
                "forces": forces,
                "iterations": item.iterations,
                "fock_builds": item.fock_builds,
                "warm_start_used": item.warm_start_used,
                "warm_start_fallback": item.warm_start_fallback,
                "executed_backend": item.executed_backend,
                "force_route": _force_route(owner, plan["arm"]),
                "oracle_errors": errors,
            }
            record["clean_ef"].append(sample)
            _save(arguments.output, record)
            if any(errors[key] > limit for key, limit in gates.items()):
                raise RuntimeError("independent E/F oracle gate failed")
            if repeat == 0:
                owner.set_warm_start_updates(False)
        record["status"] = "PASS_SCOPED_COMPLETE_ENDPOINTS"
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
        if owner is not None:
            owner.close()
        _save(arguments.output, record)
    print(json.dumps({"status": record["status"], "output": str(arguments.output)}))
    if record["status"] != "PASS_SCOPED_COMPLETE_ENDPOINTS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
