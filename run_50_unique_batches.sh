#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
GENERATOR="examples/unicycle/start_to_goal/generate_operational_rule_datasets.py"

N_BATCHES="${N_BATCHES:-50}"
SAMPLES_PER_BATCH="${SAMPLES_PER_BATCH:-50}"
BASE_OUT_DIR="${BASE_OUT_DIR:-examples/unicycle/start_to_goal/results/unique_50_batches}"
SEED_STRIDE="${SEED_STRIDE:-100000}"
START_BATCH_INDEX="${START_BATCH_INDEX:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
USE_SAFE_RUNTIME="${USE_SAFE_RUNTIME:-0}"
FAILURE_FOCUSED="${FAILURE_FOCUSED:-0}"
TEE_OUTPUT="${TEE_OUTPUT:-0}"

DT="${DT:-0.5}"
TF="${TF:-15.0}"
PRINT_EVERY="${PRINT_EVERY:-2}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2}"
FAILURE_FOCUS_MAX_ATTEMPTS="${FAILURE_FOCUS_MAX_ATTEMPTS:-5000}"
RANDOM_SAMPLE_PERCENTAGE="${RANDOM_SAMPLE_PERCENTAGE:-10}"
FAILURE_TARGET="${FAILURE_TARGET:-either}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=/path/to/python if you want to use another environment." >&2
  exit 1
fi

mkdir -p "${BASE_OUT_DIR}"

batch_is_complete() {
  local batch_dir="$1"
  "${PYTHON_BIN}" - "${batch_dir}" "${SAMPLES_PER_BATCH}" <<'PY'
import csv
import json
import sys
from pathlib import Path

batch_dir = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
systems = ("unicycle_dynamic_obstacle", "unicycle_static_obstacle")

for system in systems:
    system_dir = batch_dir / system
    metadata_path = system_dir / "metadata.json"
    paired_path = system_dir / "D_paired_comparison.csv"
    if not metadata_path.exists() or not paired_path.exists():
        sys.exit(1)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("generation_status") != "complete":
        sys.exit(1)
    with paired_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != expected_rows:
        sys.exit(1)

sys.exit(0)
PY
}

cleanup_memory() {
  sync || true
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    echo 3 > /proc/sys/vm/drop_caches || true
  fi
}

echo "Writing ${N_BATCHES}-batch dataset under: ${BASE_OUT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Seed stride: ${SEED_STRIDE}"
echo "Safe runtime: ${USE_SAFE_RUNTIME}"
echo "Failure focused: ${FAILURE_FOCUSED}"
echo "Tee output: ${TEE_OUTPUT}"

for relative_batch in $(seq 0 "$((N_BATCHES - 1))"); do
  batch_index=$((START_BATCH_INDEX + relative_batch))
  batch_dir="${BASE_OUT_DIR}/batch${batch_index}/samples_${SAMPLES_PER_BATCH}"
  log_path="${batch_dir}/generation.log"

  if [[ "${SKIP_COMPLETED}" == "1" ]] && batch_is_complete "${batch_dir}"; then
    echo "Skipping complete batch ${batch_index}/${START_BATCH_INDEX}..$((START_BATCH_INDEX + N_BATCHES - 1)): ${batch_dir}"
    continue
  fi

  mkdir -p "${batch_dir}"
  echo "Running batch ${batch_index} -> ${batch_dir}"

  generator_cmd=("${PYTHON_BIN}" -u "${GENERATOR}")
  if [[ "${USE_SAFE_RUNTIME}" == "1" ]]; then
    generator_cmd+=(--safe-runtime)
  fi
  if [[ "${FAILURE_FOCUSED}" == "1" ]]; then
    generator_cmd+=(
      --failure-focused
      --failure-target "${FAILURE_TARGET}"
      --failure-focus-max-attempts "${FAILURE_FOCUS_MAX_ATTEMPTS}"
      --random-sample-percentage "${RANDOM_SAMPLE_PERCENTAGE}"
    )
  fi

  full_cmd=(
    "${generator_cmd[@]}"
    --n-samples "${SAMPLES_PER_BATCH}" \
    --batch-index "${batch_index}" \
    --batch-seed-stride "${SEED_STRIDE}" \
    --paired-index-datasets \
    --out-dir "${batch_dir}" \
    --dt "${DT}" \
    --tf "${TF}" \
    --print-every "${PRINT_EVERY}" \
    --checkpoint-every "${CHECKPOINT_EVERY}"
  )

  if [[ "${TEE_OUTPUT}" == "1" ]]; then
    "${full_cmd[@]}" 2>&1 | tee "${log_path}"
  else
    "${full_cmd[@]}" > "${log_path}" 2>&1
    tail -n 8 "${log_path}"
  fi

  cleanup_memory
done

echo "Validating uniqueness and logical initial conditions..."
"${PYTHON_BIN}" - "${BASE_OUT_DIR}" "${N_BATCHES}" "${SAMPLES_PER_BATCH}" "${START_BATCH_INDEX}" <<'PY'
import csv
import sys
from pathlib import Path

base = Path(sys.argv[1])
n_batches = int(sys.argv[2])
samples_per_batch = int(sys.argv[3])
start_batch_index = int(sys.argv[4])

checks = {
    "unicycle_dynamic_obstacle": (
        [
            "initial_distance_to_obstacle",
            "initial_speed",
            "obstacle_radius",
            "obstacle_speed",
            "obstacle_heading",
        ],
        [
            "initial_distance_to_obstacle",
            "initial_speed",
            "obstacle_radius",
            "initial_heading_error",
            "obstacle_speed",
            "obstacle_heading",
        ],
    ),
    "unicycle_static_obstacle": (
        [
            "initial_distance_to_obstacle",
            "initial_speed",
            "obstacle_radius",
        ],
        [
            "initial_distance_to_obstacle",
            "initial_speed",
            "obstacle_radius",
            "initial_heading_error",
        ],
    ),
}

failures = []
for system, (projected_columns, full_columns) in checks.items():
    projected_keys = set()
    full_keys = set()
    rows_seen = 0
    files_seen = 0

    for batch_index in range(start_batch_index, start_batch_index + n_batches):
        paired_path = (
            base
            / f"batch{batch_index}"
            / f"samples_{samples_per_batch}"
            / system
            / "D_paired_comparison.csv"
        )
        if not paired_path.exists():
            failures.append(f"{system}: missing {paired_path}")
            continue

        files_seen += 1
        with paired_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if len(rows) != samples_per_batch:
            failures.append(f"{system}: {paired_path} has {len(rows)} rows, expected {samples_per_batch}")

        for row_idx, row in enumerate(rows):
            d0 = float(row["initial_distance_to_obstacle"])
            radius = float(row["obstacle_radius"])
            if d0 - radius <= 0.05:
                failures.append(
                    f"{system}: invalid clearance in {paired_path} row {row_idx}: "
                    f"distance={d0}, radius={radius}"
                )
            projected_keys.add(tuple(row[column] for column in projected_columns))
            full_keys.add(tuple(row[column] for column in full_columns))
            rows_seen += 1

    expected_rows = n_batches * samples_per_batch
    print(
        f"{system}: files={files_seen}/{n_batches}, rows={rows_seen}/{expected_rows}, "
        f"unique_projected={len(projected_keys)}/{rows_seen}, unique_full={len(full_keys)}/{rows_seen}"
    )
    if files_seen != n_batches:
        failures.append(f"{system}: expected {n_batches} files, found {files_seen}")
    if rows_seen != expected_rows:
        failures.append(f"{system}: expected {expected_rows} rows, found {rows_seen}")
    if len(projected_keys) != rows_seen:
        failures.append(
            f"{system}: duplicate projected cases found "
            f"({len(projected_keys)} unique out of {rows_seen})"
        )
    if len(full_keys) != rows_seen:
        failures.append(
            f"{system}: duplicate full operational cases found "
            f"({len(full_keys)} unique out of {rows_seen})"
        )

if failures:
    print("Validation failed:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    sys.exit(1)

print("Validation passed: all generated paired-comparison cases are unique and valid.")
PY
