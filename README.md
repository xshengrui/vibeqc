# H100/SM90 fork-only handoff

This orphan branch is a transfer capsule, **not** a production branch or a PR candidate. Do not merge it into `master`, apply the historical SM90 whitelist patch to current source, or attribute the old speedup to #2880. The source repository is `jinzhezenggroup/generativeqc`; the H100 follow-up is [issue #2884](https://github.com/jinzhezenggroup/generativeqc/issues/2884). A100/SM80 is separately owned. RTX 5090 reproduction is not required for this H100 issue.

## Source identities

- `legacy/` preserves the pre-#2880 local H100 campaign. The unpushed implementation commit is `704e7279f196e67eeb906f652e1f425f86be0b74`, with prerequisite parent `dc526e408174ce9d22049860a6dfa8349afc7a45`. `legacy/unmerged-704e7279.bundle` contains that commit for inspection only; it is not a fork branch and must not be merged unchanged. The historical scientific source was `cbd4d8074de35105717c8aac7cd1f86babbe90ec`.
- `post2880/` preserves the fixed-source H100 qualification at `a3d7e5c3deff7e7590bb4cb7479b49a72a594611` (merged #2880). Its report distinguishes clean complete E+F, compatibility-force diagnostics, the independent oracle, and unavailable attribution. These are not same-source before/after measurements of #2880.
- When this capsule was prepared, upstream `master` was `e265e0e45baa2affe839d3069afc7e064ae104b7`. Future work must fetch and freeze a new exact baseline once, then run same-source, same-H100 baseline/candidate pairs for any new production change.

## Contents and verification

- `MANIFEST.sha256` authenticates every payload byte. On a system with GNU tools, run `sha256sum -c MANIFEST.sha256` from this branch root. Do not silently replace missing or mismatched files.
- `legacy/` contains the historical raw archives, summaries, device probes, resource reports, patch and a Git bundle. The original `source-dc526e408.tar.gz` (89,899,329 bytes; SHA256 `05e8909770f9173dd55d5d708d4b8c6cfba1049c0d6e337dbb74692c8cbd7e50`) is **not** copied here because the source commit is available in upstream Git history. This capsule is not a byte-for-byte copy of every local file.
- `post2880/full-default-v1.tar.gz` has 24 independent native processes and 96 synchronized E+F endpoints. `post2880/gpu4-controls-v1.tar.gz` has 12 independent GPU4PySCF processes and 48 endpoints on a different physical H100. Read `post2880/h100-post2880-evidence.md` before comparing them.
- `scripts/benchmarks/` holds the uncommitted probe/report scripts from the post-#2880 worktree; `scripts/` also has the two Job shell wrappers. These are historical reproduction material, not current production code. Copy only reviewed scripts into a source-matched checkout, using independent output directories.

To inspect the old commit after cloning the normal source repository and fetching this handoff branch:

```sh
git bundle verify legacy/unmerged-704e7279.bundle
git fetch legacy/unmerged-704e7279.bundle refs/heads/codex/h100-pbe0-phase-20261029:refs/heads/archive/h100-pbe0-phase-20261029
```

The old `704e7279` Python Becke planner change conflicts with #2880's resource-based dispatch. Current Standard complete E+F uses native prepared-force, so the old compatibility-path kernel gain did not yield a current complete-endpoint gain. The 96-water DF Standard warm result at the post-#2880 source is 5.654 s; historical pre-#2880 66 s observations are cross-source and not attributable solely to #2880. No H100 production fix, PR, or scientific promotion is claimed by this capsule. Public availability of raw bytes does not itself complete independent scientific review.
