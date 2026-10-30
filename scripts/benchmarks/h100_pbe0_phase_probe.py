"""Split a frozen PBE0 H100 population into clean SCF and force observations.

The population plan supplies the measured source, library and independent oracle.
The separate phase calls are diagnostics; only complete E+F calls are endpoint
timings. Run in a fresh process on the assigned GPU with an unused output path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _oracle_errors(energy: float, forces: Any, reference: dict, atoms: int) -> dict:
    differences = [
        float(actual) - float(expected)
        for row, oracle_row in zip(forces, reference["forces"], strict=True)
        for actual, expected in zip(row, oracle_row, strict=True)
    ]
    if len(differences) != 3 * atoms:
        raise ValueError("force vector has the wrong shape")
    return {
        "energy_abs_per_atom_hartree": abs(energy - reference["energy"]) / atoms,
        "force_max_abs_hartree_per_bohr": max(map(abs, differences)),
        "force_rms_hartree_per_bohr": math.sqrt(
            math.fsum(value * value for value in differences) / len(differences)
        ),
    }


def _measure(cuda: Any, operation: Any) -> tuple[Any, float]:
    cuda.runtime.deviceSynchronize()
    start = perf_counter()
    result = operation()
    cuda.runtime.deviceSynchronize()
    return result, perf_counter() - start


def _write(path: Path, record: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--library-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.output.exists():
        parser.error("repeats must be positive and output must not exist")
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    if plan["arm"] not in ("standard", "fast") or plan["mode"] not in (
        "direct",
        "df",
    ):
        parser.error("a native Direct/DF population plan is required")
    base = Path(plan["task_root"])
    source = arguments.source_root or base / "sources/hf-dft-cbd4"
    library_root = arguments.library_root or base / "native/hf-cbd4"
    library = library_root / "libgenerativeqc.so"
    driver = source / "benchmarks/readme_defaults.py"
    planner = source / "python/generativeqc_compiler/method/stationary_resources.py"
    if _sha256(driver) != plan["driver_sha256"]:
        raise RuntimeError("frozen benchmark driver does not match plan")
    if _sha256(library) != plan["native_library_sha256"]:
        raise RuntimeError("frozen native library does not match plan")
    if Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve() != library.resolve():
        raise RuntimeError("runtime native library differs from the checked core")

    sys.path[:0] = [str(source), str(source / "python")]
    import cupy as cp
    import numpy as np

    from benchmarks import readme_defaults as frozen
    from benchmarks.readme_hf_scaling import scaling_cases

    properties = cp.cuda.runtime.getDeviceProperties(0)
    if cp.cuda.runtime.getDeviceCount() != 1 or (
        properties["major"],
        properties["minor"],
    ) != (9, 0):
        raise RuntimeError("plan requires one assigned SM90 device")
    atoms = scaling_cases()[f"water-{plan['atoms']}"].atoms
    if json.loads(json.dumps(atoms)) != plan["geometry_bohr"]:
        raise RuntimeError("frozen geometry does not match plan")
    reference = plan["gpu_l5_reference"]
    gates = plan["gates"]
    record = {
        "schema": "generativeqc.h100-pbe0-phase-probe.v1",
        "status": "INCOMPLETE",
        "plan_sha256": _sha256(arguments.plan),
        "probe_sha256": _sha256(Path(__file__)),
        "source_revision": plan["source_revision"],
        "source_root": str(source),
        "planner_sha256": _sha256(planner),
        "driver_sha256": plan["driver_sha256"],
        "library_sha256": plan["native_library_sha256"],
        "aot_sha256": {
            name: _sha256(library_root / name)
            for name in (
                "generativeqc_stationary_pbe0_rks.json",
                "generativeqc_stationary_pbe0_rks_spd.json",
                "libgenerativeqc_stationary_pbe0_rks.so",
                "libgenerativeqc_stationary_pbe0_rks_spd.so",
            )
        },
        "device": {
            "name": properties["name"].decode()
            if isinstance(properties["name"], bytes)
            else properties["name"],
            "compute_capability": "9.0",
            "runtime": cp.cuda.runtime.runtimeGetVersion(),
            "driver": cp.cuda.runtime.driverGetVersion(),
        },
        "arm": plan["arm"],
        "mode": plan["mode"],
        "atoms": plan["atoms"],
        "repeats": arguments.repeats,
        "clean_ef": [],
        "diagnostic_phases": [],
    }

    def checked(energy: float, forces: Any) -> dict:
        errors = _oracle_errors(energy, forces, reference, plan["atoms"])
        if any(errors[name] > limit for name, limit in gates.items()):
            raise RuntimeError(f"independent oracle gate failed: {errors}")
        return errors

    owner = None
    try:
        cp.cuda.runtime.deviceSynchronize()
        cold_started = perf_counter()
        calculator = frozen.native_calculator(
            "pbe0", plan["arm"], plan["mode"], base / "data/pbe0-frozen-inputs"
        )
        owner = calculator.prepare_batch([atoms], warm_start=True)
        cold_batch = owner.execute(strict=True, properties=("energy", "forces"))
        cp.cuda.runtime.deviceSynchronize()
        cold_seconds = perf_counter() - cold_started
        cold = cold_batch.items[0]
        if not cold.converged or cold.executed_backend != "cuda":
            raise RuntimeError("cold CUDA endpoint did not converge")
        record["clean_ef"].append(
            {
                "phase": "cold",
                "seconds": cold_seconds,
                "energy": float(cold.energy),
                "forces": np.asarray(cold.forces).tolist(),
                "iterations": cold.iterations,
                "fock_builds": cold.fock_builds,
                "oracle_errors": checked(float(cold.energy), cold.forces),
            }
        )
        owner.set_warm_start_updates(False)
        for repeat in range(arguments.repeats):
            batch, seconds = _measure(
                cp.cuda,
                lambda: owner.execute(strict=True, properties=("energy", "forces")),
            )
            item = batch.items[0]
            if not item.converged or item.executed_backend != "cuda":
                raise RuntimeError("warm CUDA endpoint did not converge")
            record["clean_ef"].append(
                {
                    "phase": "warm",
                    "repeat": repeat,
                    "seconds": seconds,
                    "energy": float(item.energy),
                    "forces": np.asarray(item.forces).tolist(),
                    "iterations": item.iterations,
                    "fock_builds": item.fock_builds,
                    "oracle_errors": checked(float(item.energy), item.forces),
                }
            )
        if plan["arm"] == "fast":
            record["phase_status"] = "UNAVAILABLE_FAST_AUDIT_OWNER_BOUNDARY"
            record["status"] = "PASS_SCOPED_ENDPOINT_ONLY"
        else:
            for repeat in range(arguments.repeats):
                energy_batch, scf_seconds = _measure(
                    cp.cuda,
                    lambda: owner.execute(strict=True, properties=("energy",)),
                )
                energy_item = energy_batch.items[0]
                (forces, work), force_seconds = _measure(
                    cp.cuda, lambda: owner._public_dft_cuda_force(0, atoms)
                )
                if not energy_item.converged:
                    raise RuntimeError("diagnostic SCF did not converge")
                force_array = np.asarray(forces)
                record["diagnostic_phases"].append(
                    {
                        "repeat": repeat,
                        "scf_seconds": scf_seconds,
                        "force_seconds": force_seconds,
                        "energy": float(energy_item.energy),
                        "forces": force_array.tolist(),
                        "iterations": energy_item.iterations,
                        "fock_builds": energy_item.fock_builds,
                        "oracle_errors": checked(
                            float(energy_item.energy), force_array
                        ),
                        "force_work": {
                            name: value
                            for name, value in work.items()
                            if name.startswith(("becke_", "phased_becke_", "df_"))
                            and isinstance(value, (bool, int, float, str, type(None)))
                        },
                    }
                )
            record["phase_status"] = "MEASURED_FIXED_FINAL_STATE"
            record["status"] = "PASS_SCOPED_PHASE_DIAGNOSTIC"
    except (
        ArithmeticError,
        AttributeError,
        AssertionError,
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
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        _write(arguments.output, record)
    print(json.dumps({"status": record["status"], "output": str(arguments.output)}))
    if record["status"] not in (
        "PASS_SCOPED_PHASE_DIAGNOSTIC",
        "PASS_SCOPED_ENDPOINT_ONLY",
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
