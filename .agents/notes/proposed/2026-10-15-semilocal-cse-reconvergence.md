# Decision: bound semilocal CSE reconvergence evidence before admission

Status: proposed
Date: 2026-10-15

## Problem

Issue #2507 follows the historical #2464 r2SCAN-3c direct/96 moved
observation: complete E+analytic F 296.814757 -> 403.782686 seconds,
14 -> 23 Fock builds. The force component improved 89.502930 ->
86.971727 seconds. Candidate iteration 14 had residual 1.313e-13 but
|delta E| 5.061e-12, above the unchanged 1e-12 stopping condition.
Both full populations timed out at 1800 seconds (11/12 and 10/12 returned
calls); neither has a complete moved-warm median. Direct/96 warm was
103.830 -> 100.647 seconds. These are historical observations from
0e6dfb453 versus 786c31a48, not current-source qualification.

## Bounded experiment

Freeze merged source e7c863d96c300ca3e17733532e7224ec3a19a02d before formal
native qualification. Initial preparation at df4bee32a45f1c69bbef9210ca2f58455230a720
produced host/Libxc point controls only; its receipts remain separately identified.
The integration includes merged #2511 force-consumer lowering, so neither arm
may reuse an earlier-source library or call its controls current-source results.
The tracked
experimental patch applies only in isolated source snapshots and enables the
existing ScalarCEmitter coalesce_branches API for FP64 semilocal equal-spin
emission. It changes no production default, thresholds, grid or force policy.
FP32 and polarized fallback emitters retain their existing settings.

Preregister three independent process pairs, in baseline/candidate,
candidate/baseline, baseline/candidate order. Each fresh native process performs
its own cold preparation and one moved reconvergence; no frozen-density warm
repeats substitute for independent runs. Retain the actual post-cold density
and Fock snapshot used to seed that process's moved solve. Seeds may differ
across arms; such differences must remain visible and cannot be normalized
away. NativeKS snapshot export and digesting happen outside endpoint timers.

Use the existing README direct/96 r2SCAN-3c FP64 complete endpoint with
def2-mTZVPP, grid 48/16/32, 1e-12 energy and 1e-10 density thresholds,
1e-12 screening, 100 iterations, and independent 1e-8 Eh / 1e-7 Eh/Bohr
energy/maximum-force gates. Each cold+moved process has a total 1800-second
ceiling, including preparation and evidence collection; timeout retains all
already returned observations. GPU allocation totals, including oracle,
controls, draining and retention, are limited to four card-hours. CPU-only
preparation and builds total at most three allocation-hours across all instances.

## Evidence integrity

The runner writes an exclusive preregistration and uses fresh POSIX process
groups. Timeout or failure keeps its original private scientific logs and
journal; no automatic retry is performed. The reducer reopens source
inventories, installed library bytes, patch, basis/reference bytes, native
journals and density seeds. It recomputes independent errors from full arrays,
rejects mismatched identities and duplicate/reordered/incomplete records, and
reports missing/failed/timed-out runs as INCOMPLETE. There is no successful
median for an incomplete or numerically failed population. Per-iteration
physical energies, delta E, residuals and total Fock work remain unreduced.

Recompute every returned physical gate even when the process later fails or
times out. A known numerical FAIL remains FAIL with execution completeness
INCOMPLETE; the failed process must not obscure an already observed physical
gate failure. A completed receipt additionally needs an integer zero exit,
terminal measured journal and loaded-build proof. Bind independent-reference
consumer files and actual provider versions before trusting reference arrays.

First diverging full Fock matrices remain UNAVAILABLE because the existing
solver does not export each iteration's matrix. Do not modify shared SCF to
obtain them in this slice. Three pairs support only a small-sample observation;
neither absent nor recurring tails establish unique causation.

## Current validation boundary

Host tests exercise hand-computed force/energy gates, receipt ordering,
failure/timeout retention, seed requirements and actual file-tampering
rejection. POSIX process-group and real GPU/native/oracle execution require
their own explicit evidence; Windows skips are not that evidence.
No installed library, GPU endpoint result or scientific acceptance is claimed
by this preparation commit.

## Revisit when

Only source-matched independent observations can inform later numerical
reproducibility or admission decisions. Full direct/DF 48/96 cold + five warm
+ moved + five moved-warm qualification remains outside this slice. Issues
#2507, #1117 and #1423 remain open. Historical negative evidence remains in
pinned ancestor 2f5b3396cbcfea548c76ff59d0bab91ef96ab9a9.
