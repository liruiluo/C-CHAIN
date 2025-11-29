#!/usr/bin/env bash
set -euo pipefail

# Run C-CHAIN PPO in a continual Meta-World chain:
# one agent trains sequentially across tasks in a single process.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOTAL_STEPS=1000000    # per task
GPU_NO="${GPU_NO:-0}"
SEEDS=(1 2 3 4 5)

# Default 10 tasks; override via TASKS env var (comma-separated) if needed.
TASKS_DEFAULT=(
  hammer-v3-goal-observable
  push-wall-v3-goal-observable
  faucet-close-v3-goal-observable
  push-back-v3-goal-observable
  stick-pull-v3-goal-observable
  handle-press-side-v3-goal-observable
  push-v3-goal-observable
  shelf-place-v3-goal-observable
  window-close-v3-goal-observable
  peg-unplug-side-v3-goal-observable
)

if [[ -n "${TASKS:-}" ]]; then
  IFS=',' read -r -a TASKS_ARR <<< "${TASKS}"
else
  TASKS_ARR=("${TASKS_DEFAULT[@]}")
fi
for seed in "${SEEDS[@]}"; do
  echo "=== Continual chain, seed ${seed}, total_steps_per_task=${TOTAL_STEPS} ==="
  uv run python crl_metaworld/crl_run_ppo_c_chain_metaworld_crl.py \
    --tasks "${TASKS_ARR[@]}" \
    --total_timesteps "${TOTAL_STEPS}" \
    --seed "${seed}" \
    --gpu_no "${GPU_NO}"
done
