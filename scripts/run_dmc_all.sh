#!/usr/bin/env bash
set -euo pipefail

# Run only C-CHAIN PPO on DMC continual tasks.
#
# Usage:
#   TOTAL_STEPS=1000000 GPU_NO=0 bash scripts/run_dmc_all.sh
#
# Env vars:
#   TOTAL_STEPS  - total timesteps per run (default: 1_000_000)
#   GPU_NO       - GPU index passed to --gpu_no (default: 0)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
GPU_NO="${GPU_NO:-0}"

ENV_IDS=(dog walker quadruped)
SEEDS=(1 2 3 4 5)

for env_id in "${ENV_IDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== C-CHAIN PPO | env_id=${env_id} | seed=${seed} | total_steps=${TOTAL_STEPS} ==="
    uv run python crl_dmc/crl_run_ppo_c_chain_dmc.py \
      --env_id "${env_id}" \
      --seed "${seed}" \
      --total_timesteps "${TOTAL_STEPS}" \
      --gpu_no "${GPU_NO}"
  done
done
