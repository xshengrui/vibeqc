#!/usr/bin/env bash
set -euo pipefail

task=/inspire/hdd/global_user/czxs25220150/projects/GenerativeQC/h100-portability/20261029-h100-sm90-post2880
base=/inspire/hdd/global_user/czxs25220150/projects/GenerativeQC/h100-portability/20261027-fd06
run="$task/runs/full-default-v1"
status="$task/full-default-status.txt"

test "$(cat "$task/build-status-v8.txt")" = PASS
test "$(cat "$task/runs/route-pilot-v1/status.txt")" = PASS
test ! -e "$run"
test ! -e "$status"
printf 'INCOMPLETE\n' > "$status"
trap 'result=$?; if [ "$result" -ne 0 ]; then printf "FAIL %s\n" "$result" > "$status"; fi' EXIT

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SCCACHE_DIR="$base/cache/sccache"
export LD_LIBRARY_PATH="$task/build-sm90-qualified:${LD_LIBRARY_PATH:-}"
"$base/env/bin/sccache" --show-stats > "$task/full-default-sccache-before.txt"

"$base/env/bin/python" "$task/h100_sm90_endpoint_campaign.py" \
  --base "$base" --task "$task" --output "$run" \
  --max-seconds 5400 --population-seconds 600 \
  > "$task/full-default.stdout" 2> "$task/full-default.stderr"

"$base/env/bin/python" "$task/h100_sm90_endpoint_report.py" \
  "$run" --output "$task/full-default-summary.json" \
  > "$task/full-default-report.stdout" 2> "$task/full-default-report.stderr"

"$base/env/bin/sccache" --show-stats > "$task/full-default-sccache-after.txt"
tar -czf "$task/full-default-v1.tar.gz" -C "$task/runs" full-default-v1
sha256sum "$task/full-default-v1.tar.gz" "$task/full-default-summary.json" > "$task/full-default-artifacts.sha256"
printf 'PASS\n' > "$status"
echo 'post2880-pbe0-full-default-pass'
