#!/usr/bin/env bash
set -euo pipefail

task=/inspire/hdd/global_user/czxs25220150/projects/GenerativeQC/h100-portability/20261029-h100-sm90-post2880
base=/inspire/hdd/global_user/czxs25220150/projects/GenerativeQC/h100-portability/20261027-fd06
run="$task/runs/gpu4-controls-v1"
status="$task/gpu4-controls-status.txt"

test "$(cat "$task/build-status-v8.txt")" = PASS
test "$(cat "$task/full-default-status.txt")" = PASS
test ! -e "$run"
test ! -e "$status"
printf 'INCOMPLETE\n' > "$status"
trap 'result=$?; if [ "$result" -ne 0 ]; then printf "FAIL %s\n" "$result" > "$status"; fi' EXIT

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

"$base/env/bin/python" "$task/h100_sm90_gpu4_campaign.py" \
  --base "$base" --task "$task" --output "$run" \
  --max-seconds 3600 --population-seconds 600 \
  > "$task/gpu4-controls.stdout" 2> "$task/gpu4-controls.stderr"

tar -czf "$task/gpu4-controls-v1.tar.gz" -C "$task/runs" gpu4-controls-v1
sha256sum "$task/gpu4-controls-v1.tar.gz" > "$task/gpu4-controls-artifacts.sha256"
printf 'PASS\n' > "$status"
echo 'post2880-gpu4-control-pass'
