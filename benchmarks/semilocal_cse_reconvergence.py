"""Bounded issue-2507 experiment planning and independent receipt reduction.

No production admission is changed here. The plan is written before execution;
each arm/repeat uses a fresh process and retains cold preparation and moved SCF.
Missing, failed and timed-out observations never acquire a successful median.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

from benchmarks._retention import raw_output_path

SCHEMA = "generativeqc.semilocal-cse-reconvergence.v1"
ARMS = ("baseline", "candidate")
ENERGY_GATE = 1e-8
FORCE_GATE = 1e-7
REFERENCE_CONSUMERS = (
    "benchmarks/readme_omol25.py",
    "benchmarks/readme_dft.py",
    "benchmarks/readme_wb97mv.py",
    "tools/generativeqc_gcp/reference.py",
    "tools/parameters/r2scan3c_gcp.json",
)


def digest(path: Path) -> str:
    """Hash retained bytes, streaming large native libraries."""
    if path.name in (".env", "auth.json"):
        raise ValueError("credential files cannot be evidence inputs")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    """Reject duplicate keys and nonfinite literals before interpreting evidence."""
    if path.name in (".env", "auth.json"):
        raise ValueError("credential files cannot be evidence inputs")

    def unique(pairs: list) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ValueError(f"nonfinite JSON: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=invalid,
    )


def save_new(path: Path, value: dict) -> None:
    """Exclusive creation protects the preregistration and previous failures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def make_plan(source: str, identities: dict, input_sha256: str) -> dict:
    """Preregister three paired independent owners in alternating order."""
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("an exact source commit is required")
    if set(identities) != set(ARMS):
        raise ValueError("exactly baseline and candidate identities required")
    for identity in identities.values():
        for key in ("source_identity", "library_sha256", "patch_sha256"):
            value = identity[key]
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"invalid {key}")
    if len(input_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in input_sha256
    ):
        raise ValueError("invalid input digest")
    return {
        "schema": SCHEMA,
        "source_commit": source,
        "identities": identities,
        "input_sha256": input_sha256,
        "provider": "direct",
        "atoms": 96,
        "method": "r2scan-3c",
        "precision": "fp64",
        "grid": [48, 16, 32],
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "energy_gate": ENERGY_GATE,
        "force_gate": FORCE_GATE,
        "endpoint_limit_seconds": 1800,
        "gpu_budget_seconds": 14400,
        "runs": [
            {
                "run_id": f"{arm}-{pair}",
                "pair": pair,
                "arm": arm,
                "owner": f"independent-process-{arm}-{pair}",
            }
            for pair in range(3)
            for arm in (ARMS if pair % 2 == 0 else ARMS[::-1])
        ],
    }


def finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def physical_gate(row: dict, oracle: dict) -> dict:
    """Recompute both gates from full independent arrays, never stored booleans."""
    energy, forces = row.get("energy"), row.get("forces")
    reference_energy, reference_forces = oracle.get("energy"), oracle.get("forces")
    if not finite(energy) or not finite(reference_energy):
        raise ValueError("missing/nonfinite energy")
    if forces is None or reference_forces is None:
        raise ValueError("missing force arrays")
    if not isinstance(forces, list) or not isinstance(reference_forces, list):
        raise TypeError("force arrays must be lists")
    if len(forces) != 96 or len(reference_forces) != 96:
        raise ValueError("expected all 96 force vectors")
    differences = []
    for actual, expected in zip(forces, reference_forces, strict=True):
        if len(actual) != 3 or len(expected) != 3:
            raise ValueError("invalid force vector shape")
        for a, b in zip(actual, expected, strict=True):
            if not finite(a) or not finite(b):
                raise ValueError("nonfinite force")
            differences.append(abs(a - b))
    de, df = abs(energy - reference_energy), max(differences)
    return {
        "energy_error": de,
        "force_error": df,
        "gate": de < ENERGY_GATE and df < FORCE_GATE,
    }


def validate_history(row: dict) -> None:
    """Require ordered complete physical history without normalizing work."""
    diagnostic = row.get("native_ks_diagnostic") or {}
    history = diagnostic.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("missing SCF history")
    if [h.get("iteration") for h in history] != list(range(1, len(history) + 1)):
        raise ValueError("missing, duplicate or reordered SCF iteration")
    if type(row.get("iterations")) is not int or row["iterations"] != len(history):
        raise ValueError("incomplete SCF tail")
    if type(row.get("fock_builds")) is not int or row["fock_builds"] < len(history):
        raise ValueError("invalid total Fock work")
    for index, h in enumerate(history):
        if not finite(h.get("physical_residual_max")) or h["physical_residual_max"] < 0:
            raise ValueError("missing physical residual")
        components = h.get("components", {})
        if set(components) != {"nuclear", "one_electron", "hartree", "xc"}:
            raise ValueError("missing physical energy components")
        if not all(finite(v) for v in components.values()):
            raise ValueError("nonfinite physical energy")
        if not (index == 0 and h.get("energy_change") is None) and not finite(
            h.get("energy_change")
        ):
            raise ValueError("missing energy change")
    if row.get("converged") is not True or row.get("status") != 0:
        raise ValueError("endpoint did not converge")


def reduce_campaign(plan: dict, receipts: list[dict], oracles: dict) -> dict:
    """Audit every preregistered run; retain all failures and absent run IDs."""
    expected_plan = make_plan(
        plan["source_commit"], plan["identities"], plan["input_sha256"]
    )
    if plan != expected_plan:
        raise ValueError("plan does not match bounded preregistration contract")
    expected_ids = [r["run_id"] for r in plan["runs"]]
    ids = [r["run_id"] for r in receipts]
    if len(set(ids)) != len(ids) or any(i not in expected_ids for i in ids):
        raise ValueError("duplicate or unexpected run ID")
    if ids != [i for i in expected_ids if i in ids]:
        raise ValueError("execution order differs from preregistration")
    if set(oracles) != {"cold", "moved"}:
        raise ValueError("both-geometry independent oracle required")
    if sum(r.get("wall_seconds", 0) for r in receipts) > plan["gpu_budget_seconds"]:
        raise ValueError("recorded work exceeds campaign budget")
    observed = []
    for spec in plan["runs"]:
        found = next((r for r in receipts if r["run_id"] == spec["run_id"]), None)
        if found is None:
            observed.append(
                {**spec, "status": "INCOMPLETE", "reason": "missing receipt"}
            )
            continue
        if (
            found.get("identity") != plan["identities"][spec["arm"]]
            or found.get("input_sha256") != plan["input_sha256"]
            or found.get("owner") != spec["owner"]
        ):
            raise ValueError("source/library/input/process identity mismatch")
        status = found.get("status")
        if status not in ("completed", "timeout", "failed", "not_run"):
            raise ValueError("unknown execution status")
        seconds = found.get("wall_seconds")
        if not finite(seconds) or seconds < 0:
            raise ValueError("invalid execution duration")
        item = {
            **spec,
            "status": "INCOMPLETE",
            "execution_status": status,
            "wall_seconds": seconds,
            "returned": found.get("records", []),
        }
        returned_checks = []
        phases = [row.get("phase") for row in item["returned"]]
        if phases != ["cold", "moved"][: len(phases)]:
            raise ValueError(
                "failed/partial journal contains reordered or duplicate phases"
            )
        for row in item["returned"]:
            try:
                returned_checks.append(
                    {"phase": row["phase"], **physical_gate(row, oracles[row["phase"]])}
                )
            except (TypeError, ValueError):
                returned_checks.append(
                    {
                        "phase": row["phase"],
                        "gate": None,
                        "accuracy_status": "UNAVAILABLE",
                    }
                )
        item["returned_checks"] = returned_checks
        if any(check["gate"] is False for check in returned_checks):
            item["status"] = "FAIL"
        if status == "completed":
            if type(found.get("exit_code")) is not int or found["exit_code"] != 0:
                raise ValueError("completed receipt lacks a successful process exit")
            rows = found.get("records", [])
            if [r.get("phase") for r in rows] != ["cold", "moved"]:
                raise ValueError(
                    "missing, duplicate, reordered or extraneous endpoint phase"
                )
            checks = []
            for row in rows:
                validate_history(row)
                for key in ("seconds", "complete_seconds"):
                    if not finite(row.get(key)) or row[key] <= 0 or row[key] > 1800:
                        raise ValueError("invalid/beyond-limit endpoint duration")
                checks.append(physical_gate(row, oracles[row["phase"]]))
            seed = rows[1].get("initial_seed")
            if not isinstance(seed, dict) or seed.get("owner") != spec["owner"]:
                raise ValueError("missing independently retained moved seed")
            if (
                rows[1].get("warm_start_used") is not True
                or rows[1].get("warm_start_fallback") is not False
            ):
                raise ValueError(
                    "moved solve did not consume the retained seed without fallback"
                )
            item.update(
                status="PASS" if all(c["gate"] for c in checks) else "FAIL",
                checks=checks,
                moved_seconds=rows[1]["complete_seconds"],
                moved_fock_builds=rows[1]["fock_builds"],
            )
        observed.append(item)
    complete = all(r.get("execution_status") == "completed" for r in observed)
    overall = (
        "FAIL"
        if any(r["status"] == "FAIL" for r in observed)
        else "INCOMPLETE"
        if not complete
        else "PASS"
        if all(r["status"] == "PASS" for r in observed)
        else "FAIL"
    )
    pairs = []
    for pair in range(3):
        rows = [r for r in observed if r["pair"] == pair]
        if all(r["status"] == "PASS" for r in rows):
            arms = {r["arm"]: r for r in rows}
            pairs.append(
                {
                    "pair": pair,
                    "extra_candidate_focks": arms["candidate"]["moved_fock_builds"]
                    - arms["baseline"]["moved_fock_builds"],
                }
            )
    return {
        "schema": SCHEMA,
        "scientific_status": overall,
        "execution_completeness": "COMPLETE" if complete else "INCOMPLETE",
        "runs": observed,
        "pairs": pairs,
        "sample_size": len(pairs),
        "moved_medians": {
            arm: median(r["moved_seconds"] for r in observed if r["arm"] == arm)
            for arm in ARMS
        }
        if overall == "PASS"
        else None,
        "first_diverging_fock": "UNAVAILABLE: full per-iteration matrices not exported",
        "default_admission": "NOT_EVALUATED",
        "full_parent_matrix": "INCOMPLETE",
    }


def audit_campaign_files(
    plan: dict, bindings: dict, directory: Path
) -> tuple[list[dict], dict]:
    """Reopen actual source, binary, journal, seed and reference bytes before reporting.

    The receipt is an index into original evidence, never authority for a stored
    success bit. Large arrays and libraries remain in ignored campaign storage.
    """
    from tools.generate_build_identity import _inventory, _source_identity

    references = []
    for arm in ARMS:
        binding = bindings[arm]
        source = Path(binding["source"])
        identity = plan["identities"][arm]
        manifest = source / "cmake/GenerativeQCSourceIdentity.json"
        if (
            _source_identity(source, _inventory(source, manifest))
            != identity["source_identity"]
        ):
            raise ValueError("source inventory changed after measurement")
        if digest(Path(binding["library"])) != identity["library_sha256"]:
            raise ValueError("library changed after measurement")
        if digest(Path(binding["patch"])) != identity["patch_sha256"]:
            raise ValueError("candidate patch changed after measurement")
        inputs = {name: digest(Path(binding[name])) for name in ("basis", "reference")}
        if (
            hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
            != plan["input_sha256"]
        ):
            raise ValueError("input/reference changed after measurement")
        reference = read_json(Path(binding["reference"]))
        if (
            reference.get("engine") != "reference"
            or reference.get("status") != "measured"
        ):
            raise ValueError("independent reference did not finish")
        if [r.get("phase") for r in reference.get("records", [])] != ["cold", "moved"]:
            raise ValueError(
                "independent reference has incomplete/reordered geometries"
            )
        references.append(reference)
    if references[0] != references[1]:
        raise ValueError("paired arms do not use identical independent references")
    audit_reference_sources(references[0], Path(bindings["baseline"]["source"]))
    protocol = references[0]["protocol"]
    expected_controls = {
        "atoms": 96,
        "method": "R2SCAN-3C/RKS",
        "basis": "def2-mTZVPP",
        "density_fitting": False,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "energy_gate": ENERGY_GATE,
        "force_gate": FORCE_GATE,
        "repeats": 0,
        "max_iterations": 100,
    }
    if any(protocol.get(k) != v for k, v in expected_controls.items()):
        raise ValueError("reference does not use the unchanged bounded protocol")
    grid = protocol.get("grid") or {}
    if any(
        grid.get(k) != v
        for k, v in {
            "radial_points": 48,
            "angular_polar": 16,
            "angular_azimuth": 32,
        }.items()
    ):
        raise ValueError("reference grid differs from preregistered grid")
    receipts = []
    for spec in plan["runs"]:
        receipt_path = directory / f"{spec['run_id']}.receipt.json"
        if not receipt_path.is_file():
            continue
        receipt = read_json(receipt_path)
        journal = directory / spec["run_id"] / "endpoint.json"
        if receipt.get("endpoint_sha256") is not None:
            if digest(journal) != receipt["endpoint_sha256"]:
                raise ValueError("endpoint journal changed after execution")
            actual = read_json(journal)
            if actual.get("records", []) != receipt.get("records", []):
                raise ValueError("receipt does not match raw endpoint arrays")
            if actual.get("protocol") != protocol:
                raise ValueError("native/reference protocol mismatch")
            build = actual.get("native_build") or {}
            if receipt.get("status") == "completed" and (
                actual.get("status") != "measured"
                or actual.get("stage") != "complete"
                or not build
            ):
                raise ValueError(
                    "completed receipt lacks terminal measured build evidence"
                )
            if build:
                identity = plan["identities"][spec["arm"]]
                if (
                    build.get("library_sha256") != identity["library_sha256"]
                    or build.get("probe", {}).get("source_identity")
                    != identity["source_identity"]
                ):
                    raise ValueError("loaded native build mismatches planned build")
            for row in actual.get("records", []):
                if row.get("phase") == "moved" and row.get("converged"):
                    seed = row.get("initial_seed") or {}
                    if seed.get("file") != "cold-seed.npz" or digest(
                        journal.parent / "cold-seed.npz"
                    ) != seed.get("sha256"):
                        raise ValueError("missing/changed moved initial density")
        elif receipt.get("status") == "completed":
            raise ValueError("completed run missing original journal")
        receipts.append(receipt)
    return receipts, {r["phase"]: r for r in references[0]["records"]}


def audit_reference_sources(reference: dict, source: Path) -> None:
    """Bind the independent oracle consumer and providers to their actual inputs."""
    hashes = reference.get("source_file_sha256") or {}
    if not set(REFERENCE_CONSUMERS).issubset(hashes):
        raise ValueError("independent reference lacks scientific consumer identities")
    packages = reference.get("environment", {}).get("packages", {})
    if any(
        not isinstance(packages.get(name), str)
        or not packages[name]
        or packages[name].lower()
        in {"unknown", "unverified", "unavailable", "not applicable"}
        for name in ("pyscf", "gpu4pyscf", "cupy", "dftd4")
    ):
        raise ValueError("independent reference lacks its actual provider versions")
    for relative, expected in hashes.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative:
            raise ValueError("reference consumer path must stay in its frozen source")
        actual = source / path
        if (
            not actual.resolve().is_relative_to(source.resolve())
            or digest(actual) != expected
        ):
            raise ValueError("independent reference consumer source changed")


def run_child(
    command: list[str], directory: Path, timeout: float, *, environment: dict
) -> dict:
    """Bound a fresh POSIX process group and preserve private logs on every exit.

    The supervisor owns this group only; it never stops a shared compiler server.
    Finite subprocess/process-group behavior is tested with real child processes.
    """
    if os.name != "posix":
        raise RuntimeError("campaign process-group supervision requires POSIX")
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 1800:
        raise ValueError("invalid endpoint process limit")
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with (
        (directory / "stdout.log").open("xb") as stdout,
        (directory / "stderr.log").open("xb") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        status = "completed"
        try:
            code = process.wait(timeout=timeout)
            if code:
                status = "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            code = process.returncode
        finally:
            # Descendant JIT workers must not survive a returned endpoint either.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return {
        "status": status,
        "exit_code": code,
        "pid": process.pid,
        "wall_seconds": time.monotonic() - start,
    }


def execute_plan(plan: dict, bindings: dict, output: Path) -> None:
    """Execute the fixed order once; no resubmission or success-only selection."""
    if plan != make_plan(
        plan["source_commit"], plan["identities"], plan["input_sha256"]
    ):
        raise ValueError("invalid preregistration")
    if os.name != "posix":
        raise RuntimeError("requires a finite POSIX GPU allocation")
    if not os.environ.get("GENERATIVEQC_QZ_WORKLOAD") and not os.environ.get(
        "SLURM_JOB_ID"
    ):
        raise RuntimeError("allocation identity required")
    allocation_started = bindings.get("allocation_started_unix")
    if (
        not finite(allocation_started)
        or not 0 <= time.time() - allocation_started < plan["gpu_budget_seconds"]
    ):
        raise ValueError("actual allocation start required; no fresh budget window")
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "plan.json", plan)
    save_new(output / "bindings.json", bindings)
    for spec in plan["runs"]:
        remaining = plan["gpu_budget_seconds"] - (time.time() - allocation_started)
        receipt = {
            **spec,
            "identity": plan["identities"][spec["arm"]],
            "input_sha256": plan["input_sha256"],
            "status": "not_run",
            "wall_seconds": 0,
            "records": [],
        }
        if remaining > 10:
            binding = bindings[spec["arm"]]
            directory = output / spec["run_id"]
            environment = dict(os.environ)
            environment.update(
                GENERATIVEQC_LIBRARY=binding["library"],
                PYTHONPATH=os.pathsep.join(
                    (binding["source"], str(Path(binding["source"]) / "python"))
                ),
            )
            command = [
                binding["python"],
                str(Path(__file__).resolve()),
                "worker",
                "--binding",
                str(output / "bindings.json"),
                "--plan",
                str(output / "plan.json"),
                "--run-id",
                spec["run_id"],
                "--output",
                str(directory / "endpoint.json"),
            ]
            receipt.update(
                run_child(
                    command,
                    directory,
                    min(1800, remaining - 10),
                    environment=environment,
                )
            )
            journal = directory / "endpoint.json"
            if journal.is_file():
                receipt["endpoint_sha256"] = digest(journal)
                try:
                    record = read_json(journal)
                    receipt.update(
                        records=record.get("records", []),
                        protocol=record.get("protocol"),
                        loaded_build=record.get("native_build"),
                    )
                except (ValueError, UnicodeError):
                    receipt.update(
                        status="failed",
                        journal_error="invalid JSON; original bytes retained",
                    )
        save_new(output / f"{spec['run_id']}.receipt.json", receipt)


def worker(plan: dict, bindings: dict, run_id: str, output: Path) -> None:
    """Observe the existing complete endpoint; hooks execute outside its timers."""
    spec = next(r for r in plan["runs"] if r["run_id"] == run_id)
    binding = bindings[spec["arm"]]
    source = Path(binding["source"])
    sys.path[:0] = [str(source), str(source / "python")]
    # This file can supervise archived source snapshots without installing it in
    # the production package or changing either arm's generated/native identity.
    import benchmarks

    benchmarks.__path__ = [str(source / "benchmarks")]
    import numpy as np
    from generativeqc._ks_snapshot import NativeKsSnapshot

    from benchmarks import readme_omol25 as endpoint
    from benchmarks.readme_dft import specification
    from tools.generate_build_identity import _inventory, _source_identity

    identity = plan["identities"][spec["arm"]]
    manifest = source / "cmake/GenerativeQCSourceIdentity.json"
    if (
        _source_identity(source, _inventory(source, manifest))
        != identity["source_identity"]
    ):
        raise ValueError("actual source inventory differs from plan")
    if digest(Path(binding["library"])) != identity["library_sha256"]:
        raise ValueError("actual native library differs from plan")
    if digest(Path(binding["patch"])) != identity["patch_sha256"]:
        raise ValueError("actual experimental patch differs from plan")
    inputs = {name: digest(Path(binding[name])) for name in ("basis", "reference")}
    if (
        hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        != plan["input_sha256"]
    ):
        raise ValueError("actual input bytes differ from plan")
    original = endpoint.native_endpoint_record
    seed = None

    def observe(item: Any, **kwargs: Any) -> dict:
        nonlocal seed
        owner = kwargs["owner"]
        library = owner._library
        import ctypes

        library.generativeqc_get_source_identity.restype = ctypes.c_char_p
        if (
            library.generativeqc_get_source_identity().decode()
            != identity["source_identity"]
        ):
            raise ValueError(
                "loaded library's compiled source identity differs from plan"
            )
        row = original(item, **kwargs)
        if kwargs["phase"] == "cold" and item.converged:
            snapshot = NativeKsSnapshot(owner, 0)
            try:
                # Native snapshot metadata gives its versioned matrix layout;
                # retain density and Fock only, avoiding a new persistent grid.
                version, n, spins, natom = snapshot.metadata[:4]
                if version not in range(1, 10) or natom != 96 or spins != 1:
                    raise ValueError("unsupported seed snapshot metadata")
                offset = 2 + 4 * natom
                count = spins * n * n
                path = output.parent / "cold-seed.npz"
                np.savez(
                    path,
                    density=snapshot.values[offset : offset + count].reshape(
                        spins, n, n
                    ),
                    fock=snapshot.values[offset + count : offset + 2 * count].reshape(
                        spins, n, n
                    ),
                    metadata=np.asarray(snapshot.metadata),
                )
                seed = {
                    "file": path.name,
                    "sha256": digest(path),
                    "owner": spec["owner"],
                    "definition": "independent native post-cold density used for moved solve",
                }
            finally:
                snapshot.close()
        if seed is not None:
            row["initial_seed"] = seed if kwargs["phase"] == "moved" else None
        return row

    endpoint.native_endpoint_record = observe
    try:
        endpoint.main(
            specification("r2scan-3c"),
            [
                "native",
                "--atoms",
                "96",
                "--basis-file",
                binding["basis"],
                "--reference",
                binding["reference"],
                "--repeats",
                "0",
                "--output",
                str(output),
            ],
        )
    finally:
        endpoint.native_endpoint_record = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--source", required=True)
    plan_parser.add_argument("--identities", type=Path, required=True)
    plan_parser.add_argument("--input-sha256", required=True)
    reduce_parser = sub.add_parser("report")
    reduce_parser.add_argument("--plan", type=Path, required=True)
    reduce_parser.add_argument("--campaign", type=Path, required=True)
    reduce_parser.add_argument("--binding", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--binding", type=Path, required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--plan", type=Path, required=True)
    worker_parser.add_argument("--binding", type=Path, required=True)
    worker_parser.add_argument("--run-id", required=True)
    for child in (plan_parser, reduce_parser, run_parser, worker_parser):
        child.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        value = make_plan(args.source, read_json(args.identities), args.input_sha256)
    elif args.command == "report":
        plan = read_json(args.plan)
        receipts, oracles = audit_campaign_files(
            plan, read_json(args.binding), args.campaign
        )
        value = reduce_campaign(plan, receipts, oracles)
    elif args.command == "run":
        execute_plan(
            read_json(args.plan), read_json(args.binding), args.output.resolve()
        )
        return
    else:
        worker(read_json(args.plan), read_json(args.binding), args.run_id, args.output)
        return
    save_new(args.output, value)


if __name__ == "__main__":
    main()
