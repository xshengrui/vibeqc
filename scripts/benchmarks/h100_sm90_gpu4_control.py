"""Run one source-bound GPU4PySCF PBE0 reference population in a finite H100 job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SOURCE = "a3d7e5c3deff7e7590bb4cb7479b49a72a594611"


def _version(*distributions: str) -> str:
    for distribution in distributions:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    raise RuntimeError(
        f"none of the required distributions is installed: {distributions}"
    )


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


def _errors(
    energy: float, forces: list[list[float]], reference: dict, atoms: int
) -> dict:
    differences = [
        actual - expected
        for row, expected_row in zip(forces, reference["forces"], strict=True)
        for actual, expected in zip(row, expected_row, strict=True)
    ]
    if len(differences) != atoms * 3:
        raise RuntimeError("reference force dimensions differ")
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
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("output must be new")
    revision = subprocess.check_output(
        ["git", "-C", str(arguments.source_root), "rev-parse", "HEAD"],
        text=True,
        timeout=30,
    ).strip()
    if revision != SOURCE:
        raise RuntimeError("source revision differs from the fixed baseline")
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    if plan["source_revision"] != SOURCE or plan["arm"] != "gpu4pyscf":
        raise RuntimeError("plan is not a source-bound GPU4PySCF population")
    driver = arguments.source_root / "benchmarks/readme_defaults.py"
    if _sha256(driver) != plan["driver_sha256"]:
        raise RuntimeError("benchmark driver differs from plan")

    sys.path[:0] = [str(arguments.source_root), str(arguments.source_root / "python")]
    import cupy as cp

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
        raise RuntimeError("one assigned SM90 GPU is required")
    atoms = scaling_cases()[f"water-{plan['atoms']}"].atoms
    if json.loads(json.dumps(atoms)) != plan["geometry_bohr"]:
        raise RuntimeError("frozen geometry differs from plan")
    stability = _errors(
        plan["gpu_l4_reference"]["energy"],
        plan["gpu_l4_reference"]["forces"],
        plan["gpu_l5_reference"],
        plan["atoms"],
    )
    if any(stability[key] > limit / 4 for key, limit in plan["gates"].items()):
        raise RuntimeError("independent oracle grid stability gate failed")

    record = {
        "schema": "generativeqc.h100-post2880-gpu4-control.v1",
        "status": "INCOMPLETE",
        "source_revision": SOURCE,
        "plan_sha256": _sha256(arguments.plan),
        "probe_sha256": _sha256(Path(__file__)),
        "driver_sha256": _sha256(driver),
        "device": {
            "name": name,
            "runtime": cp.cuda.runtime.runtimeGetVersion(),
            "driver": cp.cuda.runtime.driverGetVersion(),
        },
        "versions": {
            "pyscf": _version("pyscf"),
            "gpu4pyscf": _version("gpu4pyscf-cuda12x", "gpu4pyscf"),
            "cupy": _version("cupy-cuda12x", "cupy"),
            "numpy": _version("numpy"),
        },
        "atoms": plan["atoms"],
        "mode": plan["mode"],
        "oracle_stability": stability,
        "samples": [],
    }
    _save(arguments.output, record)

    def retain(sample: dict) -> None:
        errors = _errors(
            sample["energy"], sample["forces"], plan["gpu_l5_reference"], plan["atoms"]
        )
        sample["oracle_errors"] = errors
        record["samples"].append(sample)
        _save(arguments.output, record)
        if any(errors[key] > limit for key, limit in plan["gates"].items()):
            raise RuntimeError("GPU4PySCF reference drifted outside the frozen gates")

    options = SimpleNamespace(
        method="pbe0",
        arm="gpu4pyscf",
        mode=plan["mode"],
        inputs=Path(plan["task_root"]) / "data/pbe0-frozen-inputs",
        warm_repeats=3,
    )
    defaults.run_reference(options, atoms, record, retain)
    if len(record["samples"]) != 4:
        raise RuntimeError("incomplete GPU4PySCF cold/warm population")
    record["status"] = "PASS_SCOPED_REFERENCE_ENDPOINTS"
    _save(arguments.output, record)
    print(json.dumps({"status": record["status"], "output": str(arguments.output)}))


if __name__ == "__main__":
    main()
