#!/usr/bin/env bash
set -u
base=$1
task=$2
export SCCACHE_DIR="$base/cache/sccache"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$base/env/bin/ninja" -C "$task/build-current-dc526" -j4 \
  generativeqc \
  generativeqc_stationary_pbe0_rks_manifest \
  generativeqc_stationary_pbe0_rks_spd_manifest \
  > "$task/current-build.stdout" 2> "$task/current-build.stderr"
code=$?
"$base/env/bin/sccache" --show-stats > "$task/current-sccache-after.txt"
printf '%s\n' "$code" > "$task/current-build.exit.tmp"
mv "$task/current-build.exit.tmp" "$task/current-build.exit"
exit "$code"
