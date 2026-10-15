"""Independent small receipts test evidence semantics without a GPU runtime."""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.semilocal_cse_reconvergence import (
    audit_campaign_files,
    digest,
    make_plan,
    physical_gate,
    read_json,
    reduce_campaign,
    run_child,
    save_new,
)
from tools.generate_build_identity import _inventory, _source_identity


def fixtures() -> tuple[dict, list[dict], dict]:
    identities = {
        arm: {
            "source_identity": char * 64,
            "library_sha256": char * 64,
            "patch_sha256": char * 64,
        }
        for arm, char in (("baseline", "a"), ("candidate", "b"))
    }
    plan = make_plan("c" * 40, identities, "d" * 64)
    # Hand oracle: 96 equal vectors, no native-generated expected values.
    oracle = {"energy": -2.0, "forces": [[0.25, -0.5, 0.0] for _ in range(96)]}
    oracles = {phase: copy.deepcopy(oracle) for phase in ("cold", "moved")}
    receipts = []
    for spec in plan["runs"]:
        rows = []
        for phase in ("cold", "moved"):
            rows.append(
                {
                    **copy.deepcopy(oracle),
                    "phase": phase,
                    "seconds": 12.0,
                    "complete_seconds": 12.0,
                    "iterations": 1,
                    "fock_builds": 1,
                    "converged": True,
                    "status": 0,
                    "native_ks_diagnostic": {
                        "history": [
                            {
                                "iteration": 1,
                                "energy_change": None,
                                "physical_residual_max": 1e-13,
                                "components": {
                                    "nuclear": 1.0,
                                    "one_electron": -4.0,
                                    "hartree": 0.75,
                                    "xc": 0.25,
                                },
                            }
                        ]
                    },
                }
            )
        rows[1]["initial_seed"] = {
            "owner": spec["owner"],
            "file": "cold-seed.npz",
            "sha256": "e" * 64,
        }
        receipts.append(
            {
                **spec,
                "identity": identities[spec["arm"]],
                "input_sha256": plan["input_sha256"],
                "status": "completed",
                "wall_seconds": 24.0,
                "records": rows,
            }
        )
    return plan, receipts, oracles


class EvidenceTests(unittest.TestCase):
    def test_original_files_are_reopened_and_tampering_is_rejected(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "cmake").mkdir(parents=True)
            manifest = source / "cmake/GenerativeQCSourceIdentity.json"
            save_new(manifest, {"schema_version": 1, "files": ["mathematics.txt"]})
            (source / "mathematics.txt").write_text("x+x", encoding="utf-8")
            source_identity = _source_identity(source, _inventory(source, manifest))
            library, patch, basis, reference = [
                root / name for name in ("library", "patch", "basis", "reference")
            ]
            for path in (library, patch, basis):
                path.write_bytes(path.name.encode())
            plan, receipts, oracles = fixtures()
            protocol = {
                "atoms": 96,
                "method": "R2SCAN-3C/RKS",
                "basis": "def2-mTZVPP",
                "density_fitting": False,
                "energy_tolerance": 1e-12,
                "density_tolerance": 1e-10,
                "energy_gate": 1e-8,
                "force_gate": 1e-7,
                "repeats": 0,
                "max_iterations": 100,
            }
            save_new(
                reference,
                {
                    "engine": "reference",
                    "status": "measured",
                    "protocol": protocol,
                    "records": [{**v, "phase": k} for k, v in oracles.items()],
                },
            )
            identity = {
                "source_identity": source_identity,
                "library_sha256": digest(library),
                "patch_sha256": digest(patch),
            }
            input_hash = hashlib.sha256(
                json.dumps(
                    {"basis": digest(basis), "reference": digest(reference)},
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            plan = make_plan(
                "c" * 40,
                {arm: identity for arm in ("baseline", "candidate")},
                input_hash,
            )
            binding = {
                "source": str(source),
                "library": str(library),
                "patch": str(patch),
                "basis": str(basis),
                "reference": str(reference),
            }
            bindings = {arm: binding for arm in ("baseline", "candidate")}
            for receipt in receipts:
                receipt.update(identity=identity, input_sha256=input_hash)
                folder = root / receipt["run_id"]
                folder.mkdir()
                seed = folder / "cold-seed.npz"
                seed.write_bytes(b"retained-density")
                receipt["records"][1]["initial_seed"]["sha256"] = digest(seed)
                journal = folder / "endpoint.json"
                save_new(
                    journal,
                    {
                        "records": receipt["records"],
                        "protocol": protocol,
                        "native_build": {
                            "library_sha256": digest(library),
                            "probe": {"source_identity": source_identity},
                        },
                    },
                )
                receipt["endpoint_sha256"] = digest(journal)
                save_new(root / f"{receipt['run_id']}.receipt.json", receipt)
            actual, oracle = audit_campaign_files(plan, bindings, root)
            self.assertEqual(
                reduce_campaign(plan, actual, oracle)["scientific_status"], "PASS"
            )
            for target in (
                library,
                patch,
                basis,
                source / "mathematics.txt",
                root / "baseline-0/endpoint.json",
                root / "baseline-0/cold-seed.npz",
            ):
                original = target.read_bytes()
                target.write_bytes(original + b"changed")
                with self.assertRaises(ValueError):
                    audit_campaign_files(plan, bindings, root)
                target.write_bytes(original)

    def test_manual_gate_and_all_pairs(self) -> None:
        plan, receipts, oracles = fixtures()
        result = reduce_campaign(plan, receipts, oracles)
        self.assertEqual(result["scientific_status"], "PASS")
        self.assertEqual(result["sample_size"], 3)
        self.assertEqual(result["moved_medians"], {"baseline": 12.0, "candidate": 12.0})
        self.assertEqual(result["default_admission"], "NOT_EVALUATED")

    def test_actual_array_errors_override_recorded_success(self) -> None:
        plan, receipts, oracles = fixtures()
        row = receipts[0]["records"][1]
        row.update(gate=True, energy_error=0, force_error=0)
        row["forces"][95][2] = 2e-7
        result = reduce_campaign(plan, receipts, oracles)
        self.assertEqual(result["scientific_status"], "FAIL")
        self.assertIsNone(result["moved_medians"])
        self.assertEqual(result["runs"][0]["checks"][1]["force_error"], 2e-7)

    def test_timeout_failure_and_absent_are_visible(self) -> None:
        plan, receipts, oracles = fixtures()
        receipts[0].update(status="timeout", wall_seconds=1800)
        receipts[1].update(status="failed", records=[])
        result = reduce_campaign(plan, receipts[:-1], oracles)
        self.assertEqual(result["scientific_status"], "INCOMPLETE")
        self.assertIsNone(result["moved_medians"])
        self.assertEqual(len(result["runs"]), 6)
        self.assertEqual(result["runs"][0]["execution_status"], "timeout")
        self.assertEqual(len(result["runs"][0]["returned"]), 2)
        self.assertEqual(result["runs"][-1]["reason"], "missing receipt")

    def test_rejects_identity_order_duplicate_and_incomplete_tail(self) -> None:
        mutations = [
            lambda p, r, o: r.append(copy.deepcopy(r[0])),
            lambda p, r, o: r.reverse(),
            lambda p, r, o: r[0].update(input_sha256="f" * 64),
            lambda p, r, o: r[0].update(identity=p["identities"]["candidate"]),
            lambda p, r, o: r[0]["records"].reverse(),
            lambda p, r, o: r[0]["records"][1].update(iterations=2),
            lambda p, r, o: r[0]["records"][1]["native_ks_diagnostic"]["history"][
                0
            ].update(iteration=2),
            lambda p, r, o: r[0]["records"][1].update(forces=[[0, 0, 0]]),
            lambda p, r, o: p.update(energy_gate=1e-5),
            lambda p, r, o: o.pop("moved"),
            lambda p, r, o: r[0]["records"][1].pop("initial_seed"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                plan, receipts, oracles = fixtures()
                mutation(plan, receipts, oracles)
                with self.assertRaises(ValueError):
                    reduce_campaign(plan, receipts, oracles)

    def test_no_nonfinite_or_boolean_scientific_values(self) -> None:
        for invalid in (float("nan"), float("inf"), True, None):
            _plan, receipts, oracles = fixtures()
            receipts[0]["records"][0]["energy"] = invalid
            with self.assertRaises(ValueError):
                physical_gate(receipts[0]["records"][0], oracles["cold"])

    def test_preregistration_cannot_be_overwritten_and_json_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            save_new(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                save_new(path, {"value": 2})
            self.assertEqual(read_json(path), {"value": 1})
            for invalid in ('{"a":1,"a":2}', '{"a":NaN}'):
                path.write_text(invalid, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_json(path)
            json.loads(path.read_text())  # Python's default reader accepts NaN.

    @unittest.skipUnless(os.name == "posix", "real POSIX process groups required")
    def test_fresh_process_success_failure_and_timeout_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, code, limit, expected in (
                ("ok", "print('complete')", 2, "completed"),
                ("bad", "raise SystemExit(7)", 2, "failed"),
                ("timeout", "import time; time.sleep(30)", 0.1, "timeout"),
            ):
                result = run_child(
                    [sys.executable, "-c", code],
                    root / name,
                    limit,
                    environment=dict(os.environ),
                )
                self.assertEqual(result["status"], expected)
                self.assertTrue((root / name / "stdout.log").is_file())
                with self.assertRaises(ProcessLookupError):
                    os.killpg(result["pid"], 0)

    def test_invalid_process_limit_is_rejected_before_launch(self) -> None:
        if os.name != "posix":
            with self.assertRaises(RuntimeError):
                run_child([], Path("unused"), 1, environment={})
        else:
            for limit in (0, -1, 1801, float("nan")):
                with self.assertRaises(ValueError):
                    run_child([], Path("unused"), limit, environment={})


if __name__ == "__main__":
    unittest.main()
