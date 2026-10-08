# Decision: Keep PBE point-family lowering opt-in

Status: implemented (qualification pending)
Date: 2026-10-08

## Problem

The admitted PBE point launch still forwards a runtime semilocal-family value
into its canonical device consumer. The older #1113 PBE specialization reduced
register allocation from 255 to 148 without materially improving its measured
same-work complete endpoint. A smaller static footprint therefore cannot justify
promotion by itself.

## Decision

Resolve the prepared PBE physical or admitted signed-response entry to either
generic or family-specialized canonical point lowering. The explicit
`GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION=1` selects specialized lowering;
unset or `0` retains generic. Other families retain generic lowering. The
prepared physical entry also determines the batched point entry, so the two
cannot select different family identities. PBE0's semilocal exchange scaling
remains in the canonical consumer; exact exchange remains in J/K.

## Invariants

- No new functional formula, precision, response capability, point tile, AO
  map, reduction order, or batch admission is introduced by the switch.
- A changed geometry prepares a new owner and applies the same selected policy.
- Unsupported pairs and invalid switch values on PBE owners fail during preparation.
- Generic remains the default until a source-matched real-device comparison
  passes independent numerical and complete E+F endpoint gates.

## Evidence and limits

The initial source commit `af5e192225d6498de07b6e0d6034f9d293c21a00`
passed host generation and selector checks, including 24 remote C++ tests.
Those tests cannot establish device resource use, dynamic traffic, numerical
acceptance on CUDA, or endpoint profitability. #1113's negative complete
endpoint result belongs to its older binary and is historical context only.
The fixed-work `--point-specialization` PBE0 pair protocol records both arm
identities and requires equal tile and batch requests, independent reference
vectors, actual SCF work, and retained warm/moved preparation outside replay
timing. This note makes no speedup claim.

## Revisit when

A source-matched CUDA build shows selected entry identity, generated and device
footprint, registers/stack and dynamic local traffic, plus generic versus
specialized point GPU time, full XC time, and cold/warm/moved complete E+F with
the required independent numerical gates. A static reduction alone is
insufficient.

## References

- Issue #2072; scheduling-only sibling #2073; historical #1113 and #1125.
- `docs/developer/xc_native_cuda.md` and
  `benchmarks/pbe0_xc_tile_pairs.py`.
