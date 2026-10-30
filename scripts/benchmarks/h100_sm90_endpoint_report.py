"""Replay and authenticate the fixed-source H100 PBE0 complete-endpoint campaign."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

SOURCE = "a3d7e5c3deff7e7590bb4cb7479b49a72a594611"
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
        raise ValueError("expected three positive finite independent-process samples")
    return {
        "raw_process_seconds": values,
        "median_seconds": statistics.median(values),
        "range_seconds": [min(values), max(values)],
        "population_standard_deviation_seconds": statistics.pstdev(values),
    }


def summarize(root: Path) -> dict:
    campaign_path = root / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    expected = {
        f"{atoms}-{mode}-{arm}-r{repeat}"
        for atoms, mode, arm, repeat in itertools.product(
            (48, 96), ("direct", "df"), ("standard", "fast"), range(3)
        )
    }
    if (
        campaign["status"] != "PASS_SCOPED_COMPLETE_ENDPOINTS"
        or campaign["source_revision"] != SOURCE
        or len(campaign["runs"]) != len(expected)
        or {row["stem"] for row in campaign["runs"]} != expected
    ):
        raise ValueError("campaign is not a complete fixed-source 24-process cohort")
    original_device = campaign["device"][0].split(", ")
    if (
        len(campaign["device"]) != 1
        or "H100" not in original_device[0]
        or original_device[2] != "9.0"
    ):
        raise ValueError("campaign did not start with one SM90 H100")
    device_identity = tuple(original_device[:5])
    results = {}
    observed_memory = set()
    observed_clocks = set()
    for row in campaign["runs"]:
        if row["exit_code"] or not row["result_sha256"]:
            raise ValueError(f"failed population: {row['stem']}")
        for boundary in ("device_before", "device_after"):
            samples = row[boundary]
            if (
                len(samples) != 1
                or tuple(samples[0].split(", ")[:5]) != device_identity
            ):
                raise ValueError(f"physical H100 changed: {row['stem']}")
            fields = samples[0].split(", ")
            observed_memory.add(fields[5])
            observed_clocks.add(fields[7])
        path = root / f"{row['stem']}.json"
        if _sha256(path) != row["result_sha256"]:
            raise ValueError(f"population bytes changed: {row['stem']}")
        result = json.loads(path.read_text(encoding="utf-8"))
        atoms, mode, arm, repeat = row["stem"].split("-")
        if (
            result["status"] != "PASS_SCOPED_COMPLETE_ENDPOINTS"
            or result["source_revision"] != SOURCE
            or result["plan_sha256"] != row["plan_sha256"]
            or result["probe_sha256"] != campaign["probe_sha256"]
            or result["driver_sha256"] != campaign["driver_sha256"]
            or result["library_sha256"] != campaign["library_sha256"]
            or (result["atoms"], result["mode"], result["arm"])
            != (int(atoms), mode, arm)
            or len(result["clean_ef"]) != 4
        ):
            raise ValueError(f"source, plan, input or endpoint mismatch: {row['stem']}")
        phases = [sample["phase"] for sample in result["clean_ef"]]
        if phases != ["cold", "warm", "warm", "warm"]:
            raise ValueError(f"cold/warm sequence changed: {row['stem']}")
        for sample in result["clean_ef"]:
            if (
                not math.isfinite(sample["seconds"])
                or sample["seconds"] <= 0
                or sample["executed_backend"] != "cuda"
                or not math.isfinite(sample["energy"])
                or len(sample["forces"]) != int(atoms)
                or any(
                    len(force) != 3 or not all(map(math.isfinite, force))
                    for force in sample["forces"]
                )
                or any(
                    not math.isfinite(sample["oracle_errors"][name])
                    or sample["oracle_errors"][name] > limit
                    for name, limit in GATES.items()
                )
            ):
                raise ValueError(
                    f"invalid, non-CUDA or inaccurate endpoint: {row['stem']}"
                )
        results[(int(atoms), mode, arm, int(repeat[1:]))] = result
    aot_identities = {
        json.dumps(result["aot_sha256"], sort_keys=True) for result in results.values()
    }
    if len(aot_identities) != 1:
        raise ValueError("AOT artifacts changed within complete campaign")
    groups = []
    for atoms, mode, arm in itertools.product(
        (48, 96), ("direct", "df"), ("standard", "fast")
    ):
        processes = [results[(atoms, mode, arm, repeat)] for repeat in range(3)]
        cold_seconds = [process["clean_ef"][0]["seconds"] for process in processes]
        warm_seconds = [
            [sample["seconds"] for sample in process["clean_ef"][1:]]
            for process in processes
        ]
        warm_process_medians = [statistics.median(values) for values in warm_seconds]
        all_samples = [
            sample for process in processes for sample in process["clean_ef"]
        ]
        groups.append(
            {
                "atoms": atoms,
                "mode": mode,
                "arm": arm,
                "cold_seconds": _statistics(cold_seconds),
                "warm_seconds_by_process": warm_seconds,
                "warm_process_medians_seconds": _statistics(warm_process_medians),
                "cold_work": [
                    [
                        process["clean_ef"][0]["iterations"],
                        process["clean_ef"][0]["fock_builds"],
                    ]
                    for process in processes
                ],
                "warm_work": [
                    [
                        [sample["iterations"], sample["fock_builds"]]
                        for sample in process["clean_ef"][1:]
                    ]
                    for process in processes
                ],
                "force_routes": [
                    {
                        "native_prepared_force_eligible": sample["force_route"][
                            "native_prepared_force_eligible"
                        ],
                        "generated_force_count": sample["force_route"][
                            "generated_force_count"
                        ],
                        "status": sample["force_route"]["status"],
                    }
                    for sample in all_samples
                ],
                "maximum_oracle_errors": {
                    name: max(sample["oracle_errors"][name] for sample in all_samples)
                    for name in GATES
                },
            }
        )
    return {
        "schema": "generativeqc.h100-post2880-complete-summary.v1",
        "status": "PASS_SCOPED_COMPLETE_ENDPOINTS",
        "campaign_sha256": _sha256(campaign_path),
        "source_revision": SOURCE,
        "driver_sha256": campaign["driver_sha256"],
        "library_sha256": campaign["library_sha256"],
        "probe_sha256": campaign["probe_sha256"],
        "aot_sha256": json.loads(aot_identities.pop()),
        "device_identity": device_identity,
        "sampled_memory_used": sorted(observed_memory),
        "sampled_sm_clocks": sorted(observed_clocks),
        "independent_processes": len(results),
        "complete_ef_endpoints": len(results) * 4,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("output must be new")
    result = summarize(arguments.root)
    with arguments.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "groups": len(result["groups"])}))


if __name__ == "__main__":
    main()
