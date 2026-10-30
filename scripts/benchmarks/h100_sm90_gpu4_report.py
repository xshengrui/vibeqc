"""Replay the separate H100 GPU4PySCF complete-endpoint control cohort."""

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
        raise ValueError("expected three finite independent process times")
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
        f"{atoms}-{mode}-r{repeat}"
        for atoms, mode, repeat in itertools.product(
            (48, 96), ("direct", "df"), range(3)
        )
    }
    if (
        campaign["status"] != "PASS_SCOPED_REFERENCE_ENDPOINTS"
        or campaign["source_revision"] != SOURCE
        or len(campaign["runs"]) != 12
        or {row["stem"] for row in campaign["runs"]} != expected
    ):
        raise ValueError("GPU4PySCF cohort is incomplete or source-mismatched")
    device = campaign["device"]
    if len(device) != 1 or "H100" not in device[0]:
        raise ValueError("control cohort has no unique H100")
    identity = tuple(device[0].split(", ")[:5])
    results = {}
    versions = set()
    maximum_errors = {name: 0.0 for name in GATES}
    for row in campaign["runs"]:
        if row["exit_code"] or not row["result_sha256"]:
            raise ValueError(f"failed GPU4PySCF population: {row['stem']}")
        for boundary in ("device_before", "device_after"):
            samples = row[boundary]
            if len(samples) != 1 or tuple(samples[0].split(", ")[:5]) != identity:
                raise ValueError(f"physical GPU changed: {row['stem']}")
        result_path = root / f"{row['stem']}.json"
        if _sha256(result_path) != row["result_sha256"]:
            raise ValueError(f"GPU4PySCF result bytes changed: {row['stem']}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        atoms, mode, repeat = row["stem"].split("-")
        if (
            result["status"] != "PASS_SCOPED_REFERENCE_ENDPOINTS"
            or result["source_revision"] != SOURCE
            or result["plan_sha256"] != row["plan_sha256"]
            or result["probe_sha256"] != campaign["probe_sha256"]
            or result["driver_sha256"] != campaign["driver_sha256"]
            or (result["atoms"], result["mode"]) != (int(atoms), mode)
            or len(result["samples"]) != 4
            or [sample["phase"] for sample in result["samples"]]
            != ["cold", "warm", "warm", "warm"]
        ):
            raise ValueError(f"GPU4PySCF source, plan or input mismatch: {row['stem']}")
        for sample in result["samples"]:
            if (
                not math.isfinite(sample["seconds"])
                or sample["seconds"] <= 0
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
                raise ValueError(f"invalid GPU4PySCF E/F endpoint: {row['stem']}")
            for name in GATES:
                maximum_errors[name] = max(
                    maximum_errors[name], sample["oracle_errors"][name]
                )
        versions.add(json.dumps(result["versions"], sort_keys=True))
        results[(int(atoms), mode, int(repeat[1:]))] = result
    if len(versions) != 1:
        raise ValueError("GPU4PySCF package versions changed within cohort")
    groups = []
    for atoms, mode in itertools.product((48, 96), ("direct", "df")):
        populations = [results[(atoms, mode, repeat)] for repeat in range(3)]
        cold = [population["samples"][0]["seconds"] for population in populations]
        warm_raw = [
            [sample["seconds"] for sample in population["samples"][1:]]
            for population in populations
        ]
        warm_medians = [statistics.median(values) for values in warm_raw]
        groups.append(
            {
                "atoms": atoms,
                "mode": mode,
                "cold_seconds": _statistics(cold),
                "warm_seconds_by_process": warm_raw,
                "warm_process_medians_seconds": _statistics(warm_medians),
                "cold_work": [
                    [
                        population["samples"][0]["iterations"],
                        population["samples"][0]["effective_potential_builds"],
                    ]
                    for population in populations
                ],
                "warm_work": [
                    [
                        [sample["iterations"], sample["effective_potential_builds"]]
                        for sample in population["samples"][1:]
                    ]
                    for population in populations
                ],
            }
        )
    return {
        "schema": "generativeqc.h100-post2880-gpu4-replay.v1",
        "status": "PASS_SCOPED_REFERENCE_ENDPOINTS",
        "source_revision": SOURCE,
        "campaign_sha256": _sha256(campaign_path),
        "probe_sha256": campaign["probe_sha256"],
        "device_identity": identity,
        "versions": json.loads(versions.pop()),
        "independent_processes": len(results),
        "complete_ef_endpoints": len(results) * 4,
        "maximum_oracle_errors": maximum_errors,
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
