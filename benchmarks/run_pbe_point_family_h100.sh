#!/usr/bin/env bash
set -euo pipefail

: "${EXPECTED_HEAD:?pin the reviewed source commit}"
: "${INSPIRE_JOB_NAME:?record the scheduler job name}"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
root=/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-2072-point-family
run="$root/runs/$INSPIRE_JOB_NAME"
venv="$root/venv"
build="$repo/build/issue-2072-sm90"
mkdir -p "$run"
cd "$repo"
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git status --porcelain)"

export CUDA_PATH=/usr/local/cuda
export PATH="$venv/bin:$CUDA_PATH/bin:$PATH"
export PYTHONPATH="$repo/python:$repo"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GENERATIVEQC_LIBRARY="$build/libgenerativeqc.so"
export GENERATIVEQC_STATIONARY_CACHE="$run/jit-cache"
export SCCACHE_DIR="$root/sccache"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1

"$venv/bin/sccache" --version | tee "$run/sccache-version.txt"
nvcc --version > "$run/nvcc-version.txt"
nvidia-smi -L > "$run/device.txt"
git rev-parse HEAD > "$run/source-head.txt"
"$venv/bin/sccache" --show-stats > "$run/sccache-before.txt"
trap 'status=$?; "$venv/bin/sccache" --show-stats > "$run/sccache-after.txt"; printf "%s\n" "$status" > "$run/exit.txt"' EXIT

cmake -S "$repo" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$venv/bin/sccache" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$venv/bin/sccache" \
  -DPython3_EXECUTABLE="$venv/bin/python" \
  -DGENERATIVEQC_ENABLE_CUDA=ON \
  -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=90-real \
  -DGENERATIVEQC_AOT_PROFILE=sm_90 \
  -DGENERATIVEQC_AOT_UNIT_MODE=stable-shards \
  -DGENERATIVEQC_CUDA_FAST_COMPILE=OFF \
  -DGENERATIVEQC_CUDA_SEPARABLE_COMPILATION=OFF \
  -DGENERATIVEQC_CUDA_COMPILE_JOBS=2 \
  -DGENERATIVEQC_AOT_COMPILE_JOBS=2 \
  -DGENERATIVEQC_AOT_SPLIT_COMPILE_THREADS=2 \
  -DGENERATIVEQC_BUILD_TESTS=ON \
  -DGENERATIVEQC_BUILD_CLI=OFF > "$run/configure.log" 2>&1 || {
    tail -n 100 "$run/configure.log"; exit 1;
  }
cmake --build "$build" --parallel 8 --target generativeqc generativeqc_dft_cuda_tests \
  > "$run/build.log" 2>&1 || { tail -n 100 "$run/build.log"; exit 1; }
grep -m 3 sccache "$build/build.ninja" > "$run/launcher-check.txt"
sha256sum "$build/libgenerativeqc.so" > "$run/library.sha256"

for mode in 0 1; do
  export GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION="$mode"
  "$build/generativeqc_dft_cuda_tests" > "$run/native-$mode.log" 2>&1
  "$build/generativeqc_dft_cuda_tests" --point-batches > "$run/point-batches-$mode.log" 2>&1
  "$build/generativeqc_dft_cuda_tests" --pbe0-local-ao > "$run/pbe0-local-ao-$mode.log" 2>&1
done

gzip -dc benchmarks/results/pbe0-xc-tiles-20261006/reference-48.json.gz \
  > "$run/reference-48.json"
for atoms in 48 96; do
  if [ "$atoms" = 96 ]; then
    gzip -dc benchmarks/results/pbe0-xc-tiles-20261006/reference-96.json.gz \
      > "$run/reference-96.json"
  fi
  "$venv/bin/python" -m benchmarks.pbe0_xc_tile_pairs \
    --atoms "$atoms" \
    --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json \
    --reference "$run/reference-$atoms.json" \
    --point-specialization --feasibility --repeats 1 \
    --output "$run/feasibility-$atoms.json" > "$run/feasibility-$atoms.log" 2>&1
done

"$venv/bin/python" -m benchmarks.pbe0_xc_tile_pairs \
  --atoms 48 \
  --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json \
  --reference "$run/reference-48.json" \
  --point-specialization --repeats 5 \
  --output "$run/pairs-48.json" > "$run/pairs-48.log" 2>&1

"$venv/bin/python" -c 'import json,sys; from tools.generativeqc_validation.pbe0_xc_tiles import verify_point_family_pairs; from pathlib import Path; run=Path(sys.argv[1]); verify_point_family_pairs(json.loads((run/"pairs-48.json").read_text()), json.loads((run/"reference-48.json").read_text())); print("fixed-work PBE0 point-family pairs: PASS")' "$run" \
  > "$run/audit.log" 2>&1
