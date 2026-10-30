"""Bound native H100 force cost with matched E-only and complete E+F warm calls."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter

from h100_sm90_endpoint_probe import SOURCE, _errors, _OracleImportBlock, _save, _sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("output must be new")
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    if plan["source_revision"] != SOURCE or plan["arm"] != "standard":
        raise RuntimeError("native phase probe needs fixed-source Standard input")
    revision = subprocess.check_output(
        ["git", "-C", str(arguments.source_root), "rev-parse", "HEAD"],
        text=True,
        timeout=30,
    ).strip()
    if revision != SOURCE:
        raise RuntimeError("source revision differs from fixed baseline")
    library = arguments.library_root / "libgenerativeqc.so"
    if (
        _sha256(library) != plan["native_library_sha256"]
        or _sha256(arguments.source_root / "benchmarks/readme_defaults.py")
        != plan["driver_sha256"]
        or Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve() != library.resolve()
    ):
        raise RuntimeError("source, driver or native library differs from plan")
    sys.path[:0] = [str(arguments.source_root), str(arguments.source_root / "python")]
    sys.meta_path.insert(0, _OracleImportBlock())
    import cupy as cp
    import numpy as np
    from generativeqc.automatic_artifacts import read_automatic_artifact_diagnostic

    from benchmarks import readme_defaults as defaults
    from benchmarks.readme_hf_scaling import scaling_cases

    properties = cp.cuda.runtime.getDeviceProperties(0)
    if cp.cuda.runtime.getDeviceCount() != 1 or (
        properties["major"],
        properties["minor"],
    ) != (9, 0):
        raise RuntimeError("one assigned SM90 device is required")
    atoms = scaling_cases()[f"water-{plan['atoms']}"].atoms
    if json.loads(json.dumps(atoms)) != plan["geometry_bohr"]:
        raise RuntimeError("geometry differs from frozen plan")
    record = {
        "schema": "generativeqc.h100-post2880-native-phase-bound.v1",
        "status": "INCOMPLETE",
        "scope": "E-only versus complete E+F difference, not isolated kernel execution",
        "source_revision": SOURCE,
        "plan_sha256": _sha256(arguments.plan),
        "probe_sha256": _sha256(Path(__file__)),
        "library_sha256": _sha256(library),
        "atoms": plan["atoms"],
        "mode": plan["mode"],
        "cold": None,
        "warm_pairs": [],
    }
    _save(arguments.output, record)
    owner = None
    try:
        cp.cuda.runtime.deviceSynchronize()
        started = perf_counter()
        calculator = defaults.native_calculator(
            "pbe0",
            "standard",
            plan["mode"],
            Path(plan["task_root"]) / "data/pbe0-frozen-inputs",
        )
        owner = calculator.prepare_batch([atoms], warm_start=True)
        cold = owner.execute(strict=True, properties=("energy", "forces")).items[0]
        cp.cuda.runtime.deviceSynchronize()
        cold_seconds = perf_counter() - started
        if not cold.converged or cold.executed_backend != "cuda":
            raise RuntimeError("cold native E+F failed")
        if not owner._native_dft_force_eligible(0):
            raise RuntimeError("prepared native force was not admitted")
        cold_errors = _errors(
            float(cold.energy),
            np.asarray(cold.forces),
            plan["gpu_l5_reference"],
            plan["atoms"],
        )
        if any(cold_errors[name] > limit for name, limit in plan["gates"].items()):
            raise RuntimeError("cold native E/F oracle gate failed")
        record["cold"] = {
            "seconds": cold_seconds,
            "energy": float(cold.energy),
            "forces": np.asarray(cold.forces).tolist(),
            "iterations": cold.iterations,
            "fock_builds": cold.fock_builds,
            "oracle_errors": cold_errors,
        }
        transport = owner.ks_transport_diagnostics[0]
        previous_transport = None if transport is None else transport.to_payload()
        record["cold"]["ks_transport_cumulative"] = previous_transport
        _save(arguments.output, record)
        owner.set_warm_start_updates(False)
        for repeat in range(3):
            order = (
                ("energy", "complete") if repeat % 2 == 0 else ("complete", "energy")
            )
            pair = {"repeat": repeat, "order": order}
            record["warm_pairs"].append(pair)
            for measurement in order:
                cp.cuda.runtime.deviceSynchronize()
                started = perf_counter()
                properties_to_run = (
                    ("energy",) if measurement == "energy" else ("energy", "forces")
                )
                item = owner.execute(strict=True, properties=properties_to_run).items[0]
                cp.cuda.runtime.deviceSynchronize()
                seconds = perf_counter() - started
                if not item.converged or item.executed_backend != "cuda":
                    raise RuntimeError(f"warm {measurement} CUDA endpoint failed")
                if measurement == "complete":
                    errors = _errors(
                        float(item.energy),
                        np.asarray(item.forces),
                        plan["gpu_l5_reference"],
                        plan["atoms"],
                    )
                    if any(
                        errors[name] > limit for name, limit in plan["gates"].items()
                    ):
                        raise RuntimeError("warm complete E/F oracle gate failed")
                else:
                    errors = {
                        "energy_abs_per_atom_hartree": abs(
                            float(item.energy) - plan["gpu_l5_reference"]["energy"]
                        )
                        / plan["atoms"],
                        "force_max_abs_hartree_per_bohr": None,
                        "force_rms_hartree_per_bohr": None,
                    }
                    if (
                        errors["energy_abs_per_atom_hartree"]
                        > plan["gates"]["energy_abs_per_atom_hartree"]
                    ):
                        raise RuntimeError("warm E-only oracle energy gate failed")
                pair[measurement] = {
                    "seconds": seconds,
                    "energy": float(item.energy),
                    "forces": np.asarray(item.forces).tolist()
                    if measurement == "complete"
                    else None,
                    "iterations": item.iterations,
                    "fock_builds": item.fock_builds,
                    "oracle_errors": errors,
                }
                transport = owner.ks_transport_diagnostics[0]
                current_transport = (
                    None if transport is None else transport.to_payload()
                )
                pair[measurement]["ks_transport_delta"] = (
                    None
                    if current_transport is None or previous_transport is None
                    else {
                        name: value - previous_transport[name]
                        for name, value in current_transport.items()
                    }
                )
                previous_transport = current_transport
                _save(arguments.output, record)
            if (
                pair["energy"]["iterations"] != pair["complete"]["iterations"]
                or pair["energy"]["fock_builds"] != pair["complete"]["fock_builds"]
            ):
                raise RuntimeError(
                    "E-only and E+F SCF work differs; force subtraction is invalid"
                )
            pair["complete_minus_energy_seconds"] = (
                pair["complete"]["seconds"] - pair["energy"]["seconds"]
            )
            _save(arguments.output, record)
        try:
            record["automatic_artifact"] = read_automatic_artifact_diagnostic(owner)
        except NotImplementedError:
            record["automatic_artifact"] = {"status": "UNAVAILABLE_FOR_PBE0_PROFILE"}
        record["status"] = "PASS_SCOPED_NATIVE_PHASE_BOUND"
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
    if record["status"] != "PASS_SCOPED_NATIVE_PHASE_BOUND":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
