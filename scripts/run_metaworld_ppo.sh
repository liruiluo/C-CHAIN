#!/usr/bin/env bash
set -euo pipefail

# Run pure PPO on 10 Meta-World goal-observable tasks,
# each for 1M steps and 5 seeds, using uv.
#
# Usage:
#   GPU_NO=0 bash scripts/run_metaworld_ppo.sh
#
# Env vars:
#   TOTAL_STEPS  - total timesteps per run (default: 1_000_000)
#   GPU_NO       - GPU index passed to --gpu_no (default: 0)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
GPU_NO="${GPU_NO:-0}"

TASKS=(
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

SEEDS=(1 2 3 4 5)

for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== PURE PPO | task=${task} | seed=${seed} | total_steps=${TOTAL_STEPS} ==="
    uv run python crl_metaworld/crl_run_ppo_metaworld.py \
      --env_name "${task}" \
      --total_timesteps "${TOTAL_STEPS}" \
      --seed "${seed}" \
      --gpu_no "${GPU_NO}"
  done
done

