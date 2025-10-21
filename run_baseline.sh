#!/usr/bin/env bash
# run_baselines.sh
# Run AntMaze_UMaze, FetchPush, and FetchSlide in parallel background mode.

set -euo pipefail

# ===== GPU 设置 =====
export CUDA_VISIBLE_DEVICES=7

# ===== IDs =====
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ID="${TIMESTAMP}"             # 仅用于日志/文件名
EXP_NAME="curriculum"             # 传给 main.py 的 --exp 必须是 module_map 中的键

# ===== Base log dir =====
BASE_LOGDIR="${BASE_LOGDIR:-./logs}"

# ===== Task -> log dir =====
declare -A TASK_DIRS=(
  ["AntMaze_UMaze"]="${BASE_LOGDIR}/qwen_AntMaze_UMaze"
  ["FetchPush"]="${BASE_LOGDIR}/qwen_FetchPush"
  ["FetchSlide"]="${BASE_LOGDIR}/qwen_FetchSlide"
)

# ===== Ensure dirs =====
for task in "${!TASK_DIRS[@]}"; do
  mkdir -p "${TASK_DIRS[$task]}"
done

# ===== Track PIDs (optional) =====
PIDS=()

run_task() {
  local task="$1"
  local logdir="$2"
  local exp_name="$3"
  local run_id="$4"

  local logfile="${logdir}/train-${run_id}.log"
  echo "[spawn] ${task}  ->  log: ${logfile}"

  nohup python main.py \
    --task="${task}" \
    --exp="${exp_name}" \
    --logdir="${logdir}" \
    > "${logfile}" 2>&1 &

  local pid=$!
  echo "[spawn] ${task} PID=${pid}"
  echo "${pid}" > "${logdir}/pid.${run_id}"
  PIDS+=("${pid}")
}

# ===== Launch =====
run_task "AntMaze_UMaze" "${TASK_DIRS[AntMaze_UMaze]}" "${EXP_NAME}" "${RUN_ID}"
run_task "FetchPush"     "${TASK_DIRS[FetchPush]}"     "${EXP_NAME}" "${RUN_ID}"
run_task "FetchSlide"    "${TASK_DIRS[FetchSlide]}"    "${EXP_NAME}" "${RUN_ID}"

echo "[info] All tasks started. EXP_NAME=${EXP_NAME} RUN_ID=${RUN_ID}"
echo "[info] Log directories:"
for task in "AntMaze_UMaze" "FetchPush" "FetchSlide"; do
  echo "  - ${task}: ${TASK_DIRS[$task]}"
done
