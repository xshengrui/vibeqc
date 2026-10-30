"""Record intrusive stationary CUDA phase events for a frozen PBE0 state.

This diagnostic uses the production force function and its built-in CUDA event
metrics. Profiling changes grid staging and synchronization, so these durations
must not be added to clean E+F endpoint measurements.
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
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _errors(energy: float, forces: Any, reference: dict, atoms: int) -> dict:
    differences = [
        float(actual) - float(expected)
        for row, oracle_row in zip(forces, reference["forces"], strict=True)
        for actual, expected in zip(row, oracle_row, strict=True)
    ]
    if len(differences) != 3 * atoms:
        raise ValueError("wrong force vector length")
    return {
        "energy_abs_per_atom_hartree": abs(energy - reference["energy"]) / atoms,
        "force_max_abs_hartree_per_bohr": max(map(abs, differences)),
        "force_rms_hartree_per_bohr": math.sqrt(
            math.fsum(value * value for value in differences) / len(differences)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.repeats < 1:
        parser.error("output must be new and repeats positive")
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    if plan["arm"] != "standard" or plan["mode"] not in ("direct", "df"):
        parser.error("Standard Direct/DF plan required for fixed-state profiling")
    core = arguments.library_root / "libgenerativeqc.so"
    driver = arguments.source_root / "benchmarks/readme_defaults.py"
    if _sha256(core) != plan["native_library_sha256"]:
        raise RuntimeError("native core differs from frozen plan")
    if _sha256(driver) != plan["driver_sha256"]:
        raise RuntimeError("benchmark driver differs from frozen plan")
    if Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve() != core.resolve():
        raise RuntimeError("loaded core differs from checked core")
    sys.path[:0] = [str(arguments.source_root), str(arguments.source_root / "python")]
    import cupy as cp
    import numpy as np
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc._stationary_cuda import complete_rks_cuda_gradient_diagnostic
    from generativeqc_compiler.dft import NativeAO

    from benchmarks import readme_defaults as frozen
    from benchmarks.readme_hf_scaling import scaling_cases

    try:
        from generativeqc._cuda_force_resources import ordinary_cuda_force_limits
    except ModuleNotFoundError as error:
        if error.name != "generativeqc._cuda_force_resources":
            raise
        ordinary_cuda_force_limits = None

    atoms = scaling_cases()[f"water-{plan['atoms']}"].atoms
    if json.loads(json.dumps(atoms)) != plan["geometry_bohr"]:
        raise RuntimeError("geometry differs from frozen plan")
    properties = cp.cuda.runtime.getDeviceProperties(0)
    if cp.cuda.runtime.getDeviceCount() != 1 or (
        properties["major"],
        properties["minor"],
    ) != (9, 0):
        raise RuntimeError("one assigned SM90 device required")
    record = {
        "schema": "generativeqc.h100-pbe0-device-phases.v1",
        "status": "INCOMPLETE",
        "scope": "intrusive fixed-state CUDA events; not clean endpoint timing",
        "plan_sha256": _sha256(arguments.plan),
        "probe_sha256": _sha256(Path(__file__)),
        "source_root": str(arguments.source_root),
        "library_root": str(arguments.library_root),
        "planner_sha256": _sha256(
            arguments.source_root
            / "python/generativeqc_compiler/method/stationary_resources.py"
        ),
        "library_sha256": _sha256(core),
        "atoms": plan["atoms"],
        "mode": plan["mode"],
        "repeats": arguments.repeats,
        "samples": [],
    }
    owner = None
    try:
        calculator = frozen.native_calculator(
            "pbe0",
            "standard",
            plan["mode"],
            Path(plan["task_root"]) / "data/pbe0-frozen-inputs",
        )
        owner = calculator.prepare_batch([atoms], warm_start=True)
        cold = owner.execute(strict=True, properties=("energy", "forces")).items[0]
        if not cold.converged or cold.executed_backend != "cuda":
            raise RuntimeError("cold CUDA state unavailable")
        owner.set_warm_start_updates(False)
        for repeat in range(arguments.repeats):
            energy_item = owner.execute(strict=True, properties=("energy",)).items[0]
            if not energy_item.converged or energy_item.executed_backend != "cuda":
                raise RuntimeError("warm CUDA state unavailable")
            with NativeAO(
                atoms,
                basis=calculator._basis,
                representation=calculator._representation_name,
                charge=owner._charges[0],
                multiplicity=owner._multiplicities[0],
            ) as basis:
                state = StationaryKsState.from_native(
                    owner, basis, index=0, allow_budget_stationarity=True
                )
                try:
                    if ordinary_cuda_force_limits is None:
                        device_bytes, host_bytes = 512 << 20, 256 << 20
                    else:
                        device_bytes, host_bytes = ordinary_cuda_force_limits(
                            basis.natom,
                            density_fitted=bool(
                                getattr(state._source, "density_fitted", False)
                            ),
                            hamiltonian=state._source.hamiltonian,
                        )
                    result = complete_rks_cuda_gradient_diagnostic(
                        state,
                        basis,
                        compiler=None,
                        cache=arguments.output.parent / "stationary-cache",
                        aot_directory=arguments.library_root,
                        native_grid_library=core,
                        target=owner._stationary_cuda_target(),
                        tile_points=None,
                        max_device_bytes=device_bytes,
                        max_host_bytes=host_bytes,
                        max_grid_points=None,
                        max_grid_pair_visits=None,
                        profile_device=True,
                    )
                finally:
                    state._source.close()
            forces = -np.asarray(result.gradient)
            errors = _errors(
                float(energy_item.energy),
                forces,
                plan["gpu_l5_reference"],
                plan["atoms"],
            )
            if any(errors[name] > limit for name, limit in plan["gates"].items()):
                raise RuntimeError(f"independent oracle gate failed: {errors}")
            work = dict(result.work)
            if not work.get("device_profile_enabled"):
                raise RuntimeError("native device phase profile was not selected")
            record["samples"].append(
                {
                    "repeat": repeat,
                    "energy": float(energy_item.energy),
                    "forces": forces.tolist(),
                    "oracle_errors": errors,
                    "device_phase_ms": work["device_phase_ms"],
                    "becke_threads_per_point": work.get("becke_threads_per_point"),
                    "becke_shared_bytes": work.get("becke_shared_bytes"),
                    "phased_becke_batches": work.get("phased_becke_batches"),
                    "becke_pair_state_evaluations": work.get(
                        "becke_pair_state_evaluations"
                    ),
                }
            )
        record["status"] = "PASS_SCOPED_INTRUSIVE_DEVICE_PHASES"
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
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": record["status"], "output": str(arguments.output)}))
    if record["status"] != "PASS_SCOPED_INTRUSIVE_DEVICE_PHASES":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
