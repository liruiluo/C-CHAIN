import os
from pathlib import Path

import numpy as np


TASKS = [
    "hammer-v3-goal-observable",
    "push-wall-v3-goal-observable",
    "faucet-close-v3-goal-observable",
    "push-back-v3-goal-observable",
    "stick-pull-v3-goal-observable",
    "handle-press-side-v3-goal-observable",
    "push-v3-goal-observable",
    "shelf-place-v3-goal-observable",
    "window-close-v3-goal-observable",
    "peg-unplug-side-v3-goal-observable",
]

SEEDS = [1, 2, 3, 4, 5]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results" / "metaworld_cchain"

    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    task_means = []
    task_stds = []

    print("Per-task success rates at final evaluation:")
    print(f"{'task':40s}  {'mean':>8s}  {'std':>8s}  {'n':>3s}")

    for task in TASKS:
        rates = []
        base_name = task.replace("/", "_")
        for seed in SEEDS:
            file_path = results_dir / f"{base_name}_s{seed}.npz"
            if not file_path.is_file():
                continue
            data = np.load(file_path, allow_pickle=True)
            rates.append(float(data["success_rate"]))

        if not rates:
            print(f"{task:40s}  {'NA':>8s}  {'NA':>8s}  {0:3d}")
            continue

        rates_arr = np.asarray(rates, dtype=np.float64)
        mean = float(rates_arr.mean())
        std = float(rates_arr.std(ddof=0))

        task_means.append(mean)
        task_stds.append(std)

        print(f"{task:40s}  {mean:8.4f}  {std:8.4f}  {len(rates):3d}")

    if not task_means:
        raise SystemExit("No success_rate data found for any task.")

    means_arr = np.asarray(task_means, dtype=np.float64)
    stds_arr = np.asarray(task_stds, dtype=np.float64)

    overall_mean_of_means = float(means_arr.mean())
    overall_mean_of_stds = float(stds_arr.mean())

    print("\nSummary across 10 tasks:")
    print(f"mean_of_means_success_rate = {overall_mean_of_means:.4f}")
    print(f"mean_of_stds_success_rate  = {overall_mean_of_stds:.4f}")


if __name__ == "__main__":
    main()

