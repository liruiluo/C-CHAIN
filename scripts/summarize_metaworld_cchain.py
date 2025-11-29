import os
from pathlib import Path
import csv

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
    csv_path = results_dir / "metaworld_cchain_summary.csv"

    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    task_success_means = []
    task_success_stds = []
    task_return_means = []
    task_return_stds = []

    print("Per-task success rates at final evaluation:")
    print(f"{'task':40s}  {'succ_mean':>10s}  {'succ_std':>10s}  {'ret_mean':>10s}  {'ret_std':>10s}  {'n':>3s}")

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "task",
                "succ_mean",
                "succ_std",
                "succ_mean_pm_std",
                "ret_mean",
                "ret_std",
                "num_seeds",
            ]
        )

        for task in TASKS:
            succ_rates = []
            returns = []
            base_name = task.replace("/", "_")
            for seed in SEEDS:
                file_path = results_dir / f"{base_name}_s{seed}.npz"
                if not file_path.is_file():
                    continue
                data = np.load(file_path, allow_pickle=True)
                succ_rates.append(float(data["success_rate"]))
                returns.append(float(data["mean_return"]))

            if not succ_rates:
                print(f"{task:40s}  {'NA':>10s}  {'NA':>10s}  {'NA':>10s}  {'NA':>10s}  {0:3d}")
                writer.writerow([task, "NA", "NA", "NA", "NA", "NA", 0])
                continue

            succ_arr = np.asarray(succ_rates, dtype=np.float64)
            ret_arr = np.asarray(returns, dtype=np.float64)

            succ_mean = float(succ_arr.mean())
            succ_std = float(succ_arr.std(ddof=0))
            ret_mean = float(ret_arr.mean())
            ret_std = float(ret_arr.std(ddof=0))
            succ_mean_pm_std = f"{succ_mean:.4f}±{succ_std:.4f}"

            task_success_means.append(succ_mean)
            task_success_stds.append(succ_std)
            task_return_means.append(ret_mean)
            task_return_stds.append(ret_std)

            print(
                f"{task:40s}  "
                f"{succ_mean:10.4f}  {succ_std:10.4f}  "
                f"{ret_mean:10.4f}  {ret_std:10.4f}  "
                f"{len(succ_rates):3d}"
            )
            writer.writerow([task, succ_mean, succ_std, succ_mean_pm_std, ret_mean, ret_std, len(succ_rates)])

    if not task_success_means:
        raise SystemExit("No success_rate data found for any task.")

    succ_means_arr = np.asarray(task_success_means, dtype=np.float64)
    succ_stds_arr = np.asarray(task_success_stds, dtype=np.float64)
    ret_means_arr = np.asarray(task_return_means, dtype=np.float64)
    ret_stds_arr = np.asarray(task_return_stds, dtype=np.float64)

    overall_succ_mean_of_means = float(succ_means_arr.mean())
    overall_succ_mean_of_stds = float(succ_stds_arr.mean())
    overall_ret_mean_of_means = float(ret_means_arr.mean())
    overall_ret_mean_of_stds = float(ret_stds_arr.mean())

    print("\nSummary across 10 tasks:")
    print(f"mean_of_means_success_rate = {overall_succ_mean_of_means:.4f}")
    print(f"mean_of_stds_success_rate  = {overall_succ_mean_of_stds:.4f}")
    print(f"mean_of_means_return       = {overall_ret_mean_of_means:.4f}")
    print(f"mean_of_stds_return        = {overall_ret_mean_of_stds:.4f}")

    # Append summary rows to the CSV.
    with csv_path.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([])
        writer.writerow(
            [
                "mean_across_tasks",
                overall_succ_mean_of_means,
                overall_succ_mean_of_stds,
                f"{overall_succ_mean_of_means:.4f}±{overall_succ_mean_of_stds:.4f}",
                overall_ret_mean_of_means,
                overall_ret_mean_of_stds,
                len(task_success_means),
            ]
        )


if __name__ == "__main__":
    main()
