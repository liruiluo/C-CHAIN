#!/usr/bin/env python
"""
Summarize Meta-World continual chain runs (vanilla PPO and C-CHAIN).

Scans results/metaworld_{ppo,cchain}_crl/ for npz files saved by the CRL scripts:
  {env}_s{seed}_task{idx}.npz

Outputs per-dir CSVs:
  results/metaworld_ppo_crl_summary.csv
  results/metaworld_cchain_crl_summary.csv
with columns: task, succ_mean, succ_std, succ_mean_pm_std, ret_mean, ret_std, num_seeds.
"""
import csv
from pathlib import Path
from typing import Iterable

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

RESULT_DIRS = [
    ("metaworld_cchain_crl", "metaworld_cchain_crl_summary.csv"),
    ("metaworld_ppo_crl", "metaworld_ppo_crl_summary.csv"),
]


def summarize_dir(results_dir: Path, out_csv: Path, tasks: Iterable[str], seeds: Iterable[int]) -> None:
    print(f"Summarizing {results_dir} -> {out_csv}")

    task_success_means = []
    task_success_stds = []
    task_return_means = []
    task_return_stds = []

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
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

        for task in tasks:
            base = task.replace("/", "_")
            succ_rates = []
            returns = []
            for seed in seeds:
                # Files are saved as {env}_s{seed}_task{idx}.npz; pick the last idx if multiple.
                matches = sorted(results_dir.glob(f"{base}_s{seed}_task*.npz"))
                if not matches:
                    continue
                data = np.load(matches[-1], allow_pickle=True)
                succ_rates.append(float(data["success_rate"]))
                returns.append(float(data["mean_return"]))

            if not succ_rates:
                print(f"{task:40s}  NA")
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
                f"{task:40s}  succ={succ_mean:.4f}±{succ_std:.4f}  "
                f"ret={ret_mean:.4f}±{ret_std:.4f}  n={len(succ_rates)}"
            )
            writer.writerow([task, succ_mean, succ_std, succ_mean_pm_std, ret_mean, ret_std, len(succ_rates)])

    if not task_success_means:
        print(f"No data found under {results_dir}")
        return

    succ_means_arr = np.asarray(task_success_means, dtype=np.float64)
    succ_stds_arr = np.asarray(task_success_stds, dtype=np.float64)
    ret_means_arr = np.asarray(task_return_means, dtype=np.float64)
    ret_stds_arr = np.asarray(task_return_stds, dtype=np.float64)

    overall_succ_mean_of_means = float(succ_means_arr.mean())
    overall_succ_mean_of_stds = float(succ_stds_arr.mean())
    overall_ret_mean_of_means = float(ret_means_arr.mean())
    overall_ret_mean_of_stds = float(ret_stds_arr.mean())

    with out_csv.open("a", newline="") as f:
        writer = csv.writer(f)
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

    print(
        f"Summary -> succ_mean={overall_succ_mean_of_means:.4f}±{overall_succ_mean_of_stds:.4f}, "
        f"ret_mean={overall_ret_mean_of_means:.4f}±{overall_ret_mean_of_stds:.4f}"
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    for dir_name, csv_name in RESULT_DIRS:
        results_dir = root / "results" / dir_name
        if not results_dir.is_dir():
            print(f"Skip {results_dir}: not found")
            continue
        summarize_dir(results_dir, root / "results" / csv_name, TASKS, SEEDS)


if __name__ == "__main__":
    main()
