#!/usr/bin/env bash
set -euo pipefail

# Run DMC continual RL experiments for the ICML C-CHAIN paper:
# - Baseline PPO (shared agent across tasks)
# - C-CHAIN PPO
# - Oracle PPO (reset agent per task)
#
# Each env_id (dog, walker, quadruped) is run for multiple seeds.
# Uses uv to run inside the project virtual environment.
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
    echo "=== BASELINE PPO | env_id=${env_id} | seed=${seed} | total_steps=${TOTAL_STEPS} ==="
    uv run python crl_dmc/crl_run_ppo_dmc.py \
      --env_id "${env_id}" \
      --seed "${seed}" \
      --total_timesteps "${TOTAL_STEPS}" \
      --gpu_no "${GPU_NO}"

    echo "=== C-CHAIN PPO | env_id=${env_id} | seed=${seed} | total_steps=${TOTAL_STEPS} ==="
    uv run python crl_dmc/crl_run_ppo_c_chain_dmc.py \
      --env_id "${env_id}" \
      --seed "${seed}" \
      --total_timesteps "${TOTAL_STEPS}" \
      --gpu_no "${GPU_NO}"

    echo "=== ORACLE PPO | env_id=${env_id} | seed=${seed} | total_steps=${TOTAL_STEPS} ==="
    uv run python crl_dmc/crl_run_ppo_dmc_oracle.py \
      --env_id "${env_id}" \
      --seed "${seed}" \
      --total_timesteps "${TOTAL_STEPS}" \
      --gpu_no "${GPU_NO}"
  done
done

