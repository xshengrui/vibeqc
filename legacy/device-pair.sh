#!/usr/bin/env bash
set -euo pipefail
base=$1
task=$2
output=$3
test ! -e "$output"
mkdir -p "$output"
trap 'code=$?; printf "%s\n" "$code" > "$output/exit-code.txt"' EXIT
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SCCACHE_DIR="$base/cache/sccache"
export LD_LIBRARY_PATH="$base/debug/gpu4-loader-20261027-fd06-r6/toolkit-libnames:$base/env/lib/python3.11/site-packages/nvidia/cublas/lib:$base/env/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$base/env/lib/python3.11/site-packages/nvidia/cusolver/lib:$base/env/lib/python3.11/site-packages/nvidia/cusparse/lib:$base/env/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/opt/generativeqc-reuse/cuda-12.9/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
plan="$base/jobs/gqc-pbe0-matrix-1027-fd06-r1/populations/27-96-df-standard-r0/plan.json"
nvidia-smi --query-gpu=name,uuid,compute_cap,mig.mode.current,driver_version,memory.total,power.limit,clocks.sm,clocks.mem --format=csv,noheader > "$output/device-before.csv"
run_variant() {
  variant=$1
  repeat=$2
  if [[ "$variant" == baseline ]]; then
    source="$base/sources/hf-dft-cbd4"
    library="$base/native/hf-cbd4"
  else
    source="$task/candidate-cbd4-sm90-coop"
    library="$task/native-candidate"
  fi
  export GENERATIVEQC_LIBRARY="$library/libgenerativeqc.so"
  "$base/env/bin/python" "$task/h100_pbe0_device_phase_probe_v2.py" \
    --plan "$plan" --source-root "$source" --library-root "$library" \
    --output "$output/$variant-r$repeat.json" --repeats 3 \
    > "$output/$variant-r$repeat.stdout" 2> "$output/$variant-r$repeat.stderr"
  printf '%s-r%s complete\n' "$variant" "$repeat"
}
run_variant baseline 0
run_variant candidate 0
run_variant candidate 1
run_variant baseline 1
nvidia-smi --query-gpu=name,uuid,compute_cap,mig.mode.current,driver_version,memory.total,power.limit,clocks.sm,clocks.mem --format=csv,noheader > "$output/device-after.csv"
sha256sum "$output"/*.json > "$output/results.sha256"
