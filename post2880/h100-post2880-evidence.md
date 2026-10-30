# H100/SM90 PBE0 qualification on fixed post-#2880 source

## Scope and outcome

The source was locked once to `a3d7e5c3deff7e7590bb4cb7479b49a72a594611` (merged #2880). No source update occurred during measurement. The pre-#2880 `704e7279` branch and its raw H100 archives were not rebased, pushed or modified. The current worktree made **no production-code change** and did not touch A100 ownership.

The historical 96-water DF Standard warm 66.33 s result at source `cbd4d807` does **not** reproduce as a severe 66 s complete endpoint at `a3d7e5c3d`: current H100 median is 5.654 s. This is a cross-source observation, **not** a measured speedup attributable to #2880 alone. Intervening native-force/AOT/portable changes include #2827, #2835 and #2880. An exact-source SM90 serial/cooperative control changed the **compatibility force diagnostic** but not the clean complete E+F timing, because the current Standard production endpoint admits native prepared CUDA forces (`python/generativeqc/batch.py`, `_native_dft_force_eligible` and `execute`; `src/methods/dft_method.cpp`, `native_cuda_stationary_gradient`). Thus the old two-line SM90 Python planner patch is not an independent current-source production fix and must not be pushed over #2880/A100.

## Environment and binary

- One H100 80GB HBM3 per Job, SM90, MIG disabled, driver `595.58.03`, CUDA runtime `12.9`, NVCC `12.9.86`, GCC 12.3 host, 700 W power limit. qz `CQ-NexChem`, `分布式训练空间`, `开发区-H100-cuda13.2版本-183核`, one GPU/20 CPU/200 GiB, image `generativeqc-science-py311-cuda129:20261015`. Sampled boundary memory use was 0 MiB and SM clocks were 345/1980 MHz; these are **not sustained utilization/clock telemetry or proof of no concurrent load**.
- Release/CMake `CMAKE_CUDA_ARCHITECTURES=90`, explicit `GENERATIVEQC_AOT_PROFILE=portable_cuda`, PBE0 RKS stationary AOT, verified `sccache 0.16.0`. The core is SHA256 `624f3b1295a360cffdeceeea5c1bb31acd968e89728b4b672ce7c31ee3eab9e2`; PBE0 RKS AOT `.so` is `433433a32f465c7ba7a786287165cfa2d20e12561b794af8c45a92fb665923f5`.
- Host profile resolution returns SM90 `portable_cuda`, 0 tuned selections, versus SM120 `sm_120`, 44 tuned selections. The build-tree AOT package audit is `PASS`, with native `sm_90` cubin plus PTX and `driver_ptx_jit_required=false`. Cache-miss compile requests during the complete-endpoint Job: 0 before and 0 after. This does not prove that no cache-hit artifact was loaded.
- Current #2880 Python planner admits cooperative Becke by atom/resource bounds; its native phased admission no longer requires SM120. The clean Standard PBE0 E+F route is the native prepared-force capability, not the Python compatibility route whose Becke planner was controlled in the old patch.

## Current production E+F

Water-48/96 PBE0, spherical def2-SVP, stock Direct/DF auxiliary inputs and default grids, unchanged convergence/precision policies and independent level-5 GPU4PySCF oracle. Cold includes calculator construction, preparation, SCF and force; warm reuses one owner's immutable cold density. CUDA is synchronized at both timing boundaries. Three independent processes per group, each with one cold and three warm E+F calls; warm statistics use three **process medians** (not nine pooled calls). The full 24-process/96-endpoint Job `gqc-h100-post2880-full-1030-v1` and offline hash replay both passed. All Standard samples queried native prepared-force eligibility as true; Fast's private route query is unavailable and was not zero-filled. All warm native samples executed one iteration/one Fock build; Standard cold was nine/nine, Direct Fast cold nine/ten, DF Fast cold nine/nine.

| Atoms | Model | Policy | Cold raw process seconds | Cold median ± population SD | Warm process-median seconds | Warm median ± population SD |
| ---: | :--- | :--- | :--- | ---: | :--- | ---: |
| 48 | Direct | Standard | 16.328 / 15.956 / 16.050 | 16.050 ± 0.158 | 2.678 / 2.681 / 2.673 | 2.678 ± 0.003 |
| 48 | Direct | Fast | 7.931 / 7.911 / 7.945 | 7.931 ± 0.014 | 2.278 / 2.271 / 2.283 | 2.278 ± 0.005 |
| 48 | DF | Standard | 3.509 / 3.481 / 3.521 | 3.509 ± 0.017 | 1.230 / 1.217 / 1.229 | 1.229 ± 0.006 |
| 48 | DF | Fast | 3.563 / 3.552 / 3.544 | 3.552 ± 0.008 | 1.219 / 1.236 / 1.239 | 1.236 ± 0.009 |
| 96 | Direct | Standard | 57.442 / 57.338 / 57.357 | 57.357 ± 0.045 | 9.248 / 9.259 / 9.242 | 9.248 ± 0.007 |
| 96 | Direct | Fast | 25.312 / 25.165 / 25.260 | 25.260 ± 0.061 | 7.277 / 7.269 / 7.273 | 7.273 ± 0.003 |
| 96 | DF | Standard | 12.191 / 12.259 / 12.227 | 12.227 ± 0.028 | 5.654 / 5.648 / 5.662 | 5.654 ± 0.006 |
| 96 | DF | Fast | 12.234 / 12.243 / 12.219 | 12.234 ± 0.010 | 5.656 / 5.645 / 5.656 | 5.656 ± 0.005 |

Full unrounded cold and all nine warm seconds per group, E/F vectors, work, oracle errors and per-process device snapshots are in `full-default-v1.tar.gz` (SHA256 `5e56627cd306cef707d6b29da04ddd7ef3e383e5674a9411367ac86b3f1152ed`). `full-default-summary.json` and independent Windows replay both hash to `6499bae33040107dbb1034f360babc070444f9b7f3083476369ce22c488ad865`. The maximum E/atom, force max and force RMS errors were `2.9749148211522e-7 Eh`, `2.31467782293437e-5 Eh/Bohr` and `1.09596858368672e-5 Eh/Bohr`, versus unchanged `1e-5`, `1e-4`, `3e-5` gates.

## Route and stage diagnostics

- A separate one-process-per-case synchronized native diagnostic alternated E-only and complete E+F from the same frozen cold seed, with equal one-iteration/one-Fock work. Median seconds: 48 Direct 1.50 E-only / 2.70 E+F (1.20 incremental); 48 DF 0.13 / 1.23 (1.10); 96 Direct 5.69 / 9.31 (3.63); 96 DF 0.39 / 5.68 (5.28). These are **difference bounds**, not isolated kernel timings. Current 96 DF is still force-heavy; current 96 Direct is SCF-heavy. E-only rows have no force-oracle value (`null`), not fabricated zeros.
- The native KS transport diagnostic for warm water-96 E-only versus E+F reported 0 H2D bytes, 0 matrix D2H bytes, 248 versus 364 scalar D2H bytes, and 2 versus 4 synchronizations per call. These counters cover the KS owner, **not all native force transfers or CPU waits**. `read_automatic_artifact_diagnostic` returned `UNAVAILABLE_FOR_PBE0_PROFILE`, so named-PBE0 individual force launch attribution remains unavailable.
- Fast audit: Direct cold used seven mixed-stage Fock builds, two strict refinement builds and one post-SCF build, without fallback; warm was strict FP64 with one Fock build. DF Fast cold/warm remained strict FP64 via `strict_standard_fallback`, no mixed-stage Fock, no speedup over DF Standard. The warm Direct/DF Fast paths did not claim FP32 geometry.
- The same-source compatibility-path three-way pilot selected serial 1 thread/point, cooperative-no-phase 128, and automatic phased 128; the 96 DF complete Warm E+F stayed about 5.65–5.68 s in all three. The **separate compatibility Force diagnostic** changed roughly 19.20 → 5.85 → 4.59 s and phased batches 0 → 0 → 1054. Automatic phased logical Becke pair-state work was 4,917,212,160 versus 9,834,424,320 in the nonphased controls. Do not label this a production endpoint speedup.
- A separate four-process ABBA intrusive CUDA-event probe of that compatibility path measured 96 DF geometry-kernel process medians of 16,244.84/16,242.90 ms (serial) versus 822.90/819.27 ms (automatic phased), about 19.8× for this kernel only. Primitive derivative kernels were 36–38 ms, geometry H2D 12–13 ms, final D2H ~0.04–0.05 ms, sync wait ~10–11 ms. Static `cuobjdump` on native SM90 AOT reports `geometry_kernel<double>` REG255/STACK192, `geometry_cooperative_kernel<false,double>` REG255/STACK288, and phased `geometry_cooperative_kernel<true,double>` REG92/STACK0; actual achieved occupancy is **unmeasured**. The event probe is not a native prepared-force launch trace or complete E+F timing.

## GPU4PySCF and historical limits

GPU4PySCF `1.8.1`, PySCF `2.14.0`, CuPy `13.6.0`, NumPy `2.2.6` ran four groups in a separate finite H100 Job: 12 independent processes/48 complete E+F endpoints. Same image, group, driver, gates and frozen input/oracle, but **a different physical H100 UUID** from the native 24-process Job. Its own complete-endpoint timing was correctly synchronized. It passed oracle gates and hash-checked offline replay. The first 48-atom cold GPU4 process retained a large setup outlier; do not erase it or call cold medians compiler-free.

| Atoms | Model | GPU4 cold raw seconds | Cold median ± SD | GPU4 warm process medians | Warm median ± SD |
| ---: | :--- | :--- | ---: | :--- | ---: |
| 48 | Direct | 24.595 / 8.101 / 8.067 | 8.101 ± 7.783 | 2.260 / 2.349 / 2.246 | 2.260 ± 0.046 |
| 48 | DF | 5.683 / 2.874 / 2.837 | 2.874 ± 1.333 | 1.198 / 1.204 / 1.201 | 1.201 ± 0.003 |
| 96 | Direct | 9.927 / 9.612 / 10.074 | 9.927 ± 0.192 | 3.112 / 2.913 / 2.912 | 2.913 ± 0.094 |
| 96 | DF | 7.253 / 6.833 / 6.923 | 6.923 ± 0.180 | 3.269 / 3.316 / 3.353 | 3.316 ± 0.034 |

`gpu4-controls-v1.tar.gz` SHA256 is `c3a552356c54d6785b2adcc284e1dc272a845848e5ff4f074ec97483223989f6`. Offline replay authenticated all 12 result hashes, device continuity, versions, errors and statistics. Native-versus-GPU4 values are **system-level observations, not strict paired speedups**. Historical #2863 H100 (`cbd4d807`) 96 Direct/DF Standard warm medians 38.46/66.33 s and historical RTX 5090 single-population 10.12/11.77 s use different source/environment/protocol provenance; no hardware-only ratio or RTX 5090 nonregression claim is made. No RTX 5090 allocation was obtained in this round.

## Validation, delivery and open questions

- Linux fixed-source focused host suite: `185 passed, 1 skipped` after setting the required `CCACHE` variable to verified sccache. An earlier same suite had `181 passed, 1 skipped, 4 errors` solely because that fixture's `ccache`-named launcher prerequisite was not set; the failed output remains retained. Local Windows subset had `44 passed` for architecture/resource selection; its broader native/Slurm tests lacked local cache/Slurm prerequisites. Ruff/format and Python syntax checks pass for the task-owned benchmark scripts.
- Build preparation failures remain separate from science: two qz Job nodes did not expose the new GPFS task path; initial CMake lookup and an old venv entrypoint shebang failed before compilation; two foreground Notebook build continuations were interrupted by JupyterTerminal, and a detached incremental build was stopped when the Notebook hit platform idle recycling. The final incremental `gqc-h100-post2880-build-1030-v8` Job succeeded without resetting source/cache or dropping failed receipts.
- The one delivery-time upstream fetch found `29c68aa7a` ahead of the frozen source by CI and named-correction-qualification changes only; no measured H100 production path overlaps, so no new-source performance claim or unnecessary rerun was made. The separate SM90 implementation worktree remains owned by another writer and unmodified. Original branch `codex/h100-pbe0-phase-20261029` remains at `704e7279` and unpushed.
- **No new production fix, commit, PR, merge or Release.** The current mainline no longer exhibits the historical severe DF/Force regression, and the old planner patch does not alter its native prepared complete endpoint. The task-owned benchmark scripts are uncommitted in `codex/h100-sm90-becke-evidence-20261029`; issue #2884 remains open. The full new raw archives are local and qz task artifacts, not Git-tracked because the existing aggregate results cap has insufficient headroom. Public raw/scientific-review delivery remains `INCOMPLETE`.
- Remaining H100 work: a current-source, same-protocol RTX 5090 qualification if hardware becomes available; actual native prepared-force kernel/launch/occupancy and full force H2D/D2H attribution (Nsight Systems/Compute unavailable in the image); source-matched H100-specific Direct J/K and DF force microbenchmarks before proposing any SM90 tuning. SM80/A100 adaptation is separately owned. These limitations preclude attributing the remaining native-versus-GPU4 gap to SM90 registers, occupancy or #2880 alone.

Replay commands from the task worktree:

```text
python benchmarks/h100_sm90_endpoint_report.py C:/Users/Lenovo/.codex/worktrees/h100-sm90-becke-evidence-20261029/full-default-v1 --output <new-file.json>
python benchmarks/h100_sm90_gpu4_report.py C:/Users/Lenovo/.codex/worktrees/h100-sm90-becke-evidence-20261029/gpu4-controls-v1 --output <new-file.json>
```
