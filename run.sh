#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/opt/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/opt/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "Conda initialization script not found. Install Conda or update run.sh." >&2
  exit 1
fi

conda activate cbfkit

N_BATCHES=100
SAMPLES_PER_BATCH=50
BASE_OUT_DIR="examples/unicycle/start_to_goal/results/vanilla"

cleanup_memory() {
  # The Python process exit frees user-space memory. This adds a best-effort OS cache drop.
  sync || true
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    echo 3 > /proc/sys/vm/drop_caches || true
  fi
}

for batch_idx in $(seq 1 "${N_BATCHES}"); do
  out_dir="${BASE_OUT_DIR}/batch${batch_idx}/samples_${SAMPLES_PER_BATCH}"
  echo "Running batch ${batch_idx}/${N_BATCHES} -> ${out_dir}"

  python -u examples/unicycle/start_to_goal/generate_operational_rule_datasets.py \
    --safe-runtime \
    --n-samples "${SAMPLES_PER_BATCH}" \
    --paired-index-datasets \
    --failure-focused \
    --failure-target either \
    --failure-focus-max-attempts 5000 \
    --out-dir "${out_dir}" \
    --dt 0.5 \
    --tf 15.0 \
    --print-every 2 \
    --random-sample-percentage 10 \
    --checkpoint-every 2
  #   --cbf-controller robust \
  #   --disturbance-norm-bound 0.2

  echo "Cleaning memory after batch ${batch_idx}"
  cleanup_memory
done
