"""Audit and summarize a complete interleaved H100 PBE0 campaign offline."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

GATES = {
    "energy_abs_per_atom_hartree": 1e-5,
    "force_max_abs_hartree_per_bohr": 1e-4,
    "force_rms_hartree_per_bohr": 3e-5,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _statistics(values: list[float]) -> dict:
    if len(values) != 3 or not all(
        math.isfinite(value) and value > 0 for value in values
    ):
        raise ValueError("expected three finite positive independent process values")
    return {
        "values": values,
        "median": statistics.median(values),
        "range": [min(values), max(values)],
        "population_standard_deviation": statistics.pstdev(values),
    }


def _check_result(path: Path, atoms: int, arm: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    expected_status = (
        "PASS_SCOPED_PHASE_DIAGNOSTIC"
        if arm == "standard"
        else "PASS_SCOPED_ENDPOINT_ONLY"
    )
    if result["status"] != expected_status or len(result["clean_ef"]) != 4:
        raise ValueError(f"incomplete endpoint record: {path.name}")
    if arm == "standard" and len(result["diagnostic_phases"]) != 3:
        raise ValueError(f"missing Standard phase diagnostics: {path.name}")
    if arm == "fast" and result["diagnostic_phases"]:
        raise ValueError(f"Fast phases were not independently measured: {path.name}")
    for sample in (*result["clean_ef"], *result["diagnostic_phases"]):
        forces = sample["forces"]
        if len(forces) != atoms or any(len(row) != 3 for row in forces):
            raise ValueError(f"force shape mismatch: {path.name}")
        if not math.isfinite(sample["energy"]) or not all(
            math.isfinite(value) for row in forces for value in row
        ):
            raise ValueError(f"nonfinite energy/forces: {path.name}")
        if any(
            not math.isfinite(sample["oracle_errors"][name])
            or sample["oracle_errors"][name] > limit
            for name, limit in GATES.items()
        ):
            raise ValueError(f"independent oracle gate failed: {path.name}")
    return result


def summarize(root: Path) -> dict:
    campaign_path = root / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    expected = {
        f"{atoms}-{mode}-{arm}-r{repeat}-{variant}"
        for atoms, mode, arm, repeat, variant in itertools.product(
            (48, 96),
            ("direct", "df"),
            ("standard", "fast"),
            range(3),
            ("baseline", "candidate"),
        )
    }
    if campaign["status"] != "PASS_SCOPED_NATIVE_PAIRS" or len(campaign["runs"]) != 48:
        raise ValueError("campaign is not a complete 48-process cohort")
    actual = {row["stem"] for row in campaign["runs"]}
    if actual != expected:
        raise ValueError("duplicate or missing population")
    device_rows = campaign["device"]["rows"]
    if len(device_rows) != 1:
        raise ValueError("campaign did not retain one physical GPU")
    device_identity = tuple(device_rows[0].split(", ")[:5])
    results = {}
    for row in campaign["runs"]:
        if row["exit_code"]:
            raise ValueError(f"failed population: {row['stem']}")
        for boundary in ("device_before", "device_after"):
            observed = row[boundary]["rows"]
            if (
                len(observed) != 1
                or tuple(observed[0].split(", ")[:5]) != device_identity
            ):
                raise ValueError(f"physical GPU changed: {row['stem']}")
        name = row["stem"]
        path = root / f"{name}.json"
        if _sha256(path) != row["result_sha256"]:
            raise ValueError(f"changed population bytes: {name}")
        atoms, mode, arm, repeat, variant = name.split("-")
        result = _check_result(path, int(atoms), arm)
        control = result.get("schedule_control", {})
        expected_selection = "automatic" if variant == "candidate" else "serial"
        if (
            control.get("selection") != expected_selection
            or control.get("source_revision") != campaign["source_revision"]
            or control.get("control_sha256") != campaign["control_sha256"]
            or control.get("probe_sha256") != campaign["probe_sha256"]
        ):
            raise ValueError(f"schedule control identity mismatch: {name}")
        selected_threads = {
            call["threads_per_point"] for call in control.get("planner_calls", ())
        }
        if not selected_threads or selected_threads != (
            {128} if variant == "candidate" else {1}
        ):
            raise ValueError(f"unexpected SM90 schedule selection: {name}")
        if (result["atoms"], result["mode"], result["arm"], result["repeats"]) != (
            int(atoms),
            mode,
            arm,
            3,
        ):
            raise ValueError(f"population input mismatch: {name}")
        if result["plan_sha256"] != row["plan_sha256"]:
            raise ValueError(f"plan identity mismatch: {name}")
        if result["probe_sha256"] != campaign["probe_sha256"]:
            raise ValueError(f"probe identity mismatch: {name}")
        if variant == "candidate" and (
            result["planner_sha256"] != campaign["candidate_planner_sha256"]
        ):
            raise ValueError(f"candidate planner identity mismatch: {name}")
        results[(int(atoms), mode, arm, int(repeat[1:]), variant)] = result

    identities = {
        (item["source_revision"], item["driver_sha256"], item["library_sha256"])
        for item in results.values()
    }
    if len(identities) != 1:
        raise ValueError("source, driver or native core changed within campaign")
    source_revision, driver_sha256, library_sha256 = identities.pop()
    if (
        source_revision != campaign["source_revision"]
        or library_sha256 != campaign["library_sha256"]
    ):
        raise ValueError("campaign source or native library identity mismatch")
    for atoms, mode, arm, repeat in itertools.product(
        (48, 96), ("direct", "df"), ("standard", "fast"), range(3)
    ):
        if (
            results[(atoms, mode, arm, repeat, "baseline")]["plan_sha256"]
            != results[(atoms, mode, arm, repeat, "candidate")]["plan_sha256"]
        ):
            raise ValueError("paired population plans differ")

    groups = []
    for atoms, mode, arm in itertools.product(
        (48, 96), ("direct", "df"), ("standard", "fast")
    ):
        group = {"atoms": atoms, "mode": mode, "arm": arm}
        for variant in ("baseline", "candidate"):
            population = [
                results[(atoms, mode, arm, repeat, variant)] for repeat in range(3)
            ]
            cold = [item["clean_ef"][0]["seconds"] for item in population]
            warm_raw = [
                [sample["seconds"] for sample in item["clean_ef"][1:]]
                for item in population
            ]
            warm_medians = [statistics.median(samples) for samples in warm_raw]
            values = {
                "cold_seconds": _statistics(cold),
                "cold_work": [
                    [
                        item["clean_ef"][0]["iterations"],
                        item["clean_ef"][0]["fock_builds"],
                    ]
                    for item in population
                ],
                "warm_seconds_by_process": warm_raw,
                "warm_process_medians_seconds": _statistics(warm_medians),
                "warm_work": [
                    [
                        [sample["iterations"], sample["fock_builds"]]
                        for sample in item["clean_ef"][1:]
                    ]
                    for item in population
                ],
                "maximum_oracle_errors": {
                    name: max(
                        sample["oracle_errors"][name]
                        for item in population
                        for sample in item["clean_ef"]
                    )
                    for name in GATES
                },
                "aot_sha256": population[0]["aot_sha256"],
            }
            if any(item["aot_sha256"] != values["aot_sha256"] for item in population):
                raise ValueError("AOT artifact changed within one timing group")
            if arm == "standard":
                values["scf_seconds"] = _statistics(
                    [
                        statistics.median(
                            [
                                phase["scf_seconds"]
                                for phase in item["diagnostic_phases"]
                            ]
                        )
                        for item in population
                    ]
                )
                values["force_seconds"] = _statistics(
                    [
                        statistics.median(
                            [
                                phase["force_seconds"]
                                for phase in item["diagnostic_phases"]
                            ]
                        )
                        for item in population
                    ]
                )
                values["becke_threads_per_point"] = sorted(
                    {
                        phase["force_work"]["becke_threads_per_point"]
                        for item in population
                        for phase in item["diagnostic_phases"]
                    }
                )
                values["becke_pair_state_evaluations"] = sorted(
                    {
                        phase["force_work"]["becke_pair_state_evaluations"]
                        for item in population
                        for phase in item["diagnostic_phases"]
                    }
                )
                values["phased_becke_batches"] = sorted(
                    {
                        phase["force_work"].get("phased_becke_batches")
                        for item in population
                        for phase in item["diagnostic_phases"]
                    }
                )
            group[variant] = values
        if arm == "standard" and (
            group["baseline"]["becke_pair_state_evaluations"]
            != group["candidate"]["becke_pair_state_evaluations"]
        ):
            raise ValueError("Becke semantic work changed between schedules")
        group["cold_speedup"] = (
            group["baseline"]["cold_seconds"]["median"]
            / group["candidate"]["cold_seconds"]["median"]
        )
        group["warm_speedup"] = (
            group["baseline"]["warm_process_medians_seconds"]["median"]
            / group["candidate"]["warm_process_medians_seconds"]["median"]
        )
        groups.append(group)
    device_fields = campaign["device"]["rows"][0].split(", ")
    snapshot_fields = [
        snapshot["rows"][0].split(", ")
        for row in campaign["runs"]
        for snapshot in (row["device_before"], row["device_after"])
    ]
    return {
        "schema": "generativeqc.h100-pbe0-becke-interleaved-summary.v1",
        "status": "PASS_SCOPED_NATIVE_PAIRS",
        "schedule_variants": {"baseline": "serial", "candidate": "automatic"},
        "campaign_sha256": _sha256(campaign_path),
        "population_count": len(results),
        "processes_per_variant_group": 3,
        "warm_calls_per_process": 3,
        "device": {
            "name": device_fields[0],
            "compute_capability": device_fields[2],
            "mig": device_fields[3],
            "driver": device_fields[4],
            "memory_total": device_fields[5],
            "power_limit": device_fields[7],
            "sampled_memory_used": sorted({fields[6] for fields in snapshot_fields}),
            "sampled_sm_clocks": sorted({fields[9] for fields in snapshot_fields}),
        },
        "probe_sha256": campaign["probe_sha256"],
        "source_revision": source_revision,
        "driver_sha256": driver_sha256,
        "library_sha256": library_sha256,
        "candidate_planner_sha256": campaign["candidate_planner_sha256"],
        "gates": GATES,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("output must not exist")
    result = summarize(arguments.root)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "groups": len(result["groups"])}))


if __name__ == "__main__":
    main()
