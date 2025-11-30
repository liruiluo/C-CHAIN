#!/usr/bin/env python
"""
Summarize DMC (walker/quadruped/dog) runs for PPO and C-CHAIN PPO.

Expected result files:
- crl_dmc: no npz outputs, so we summarize from TensorBoard scalars.
- Run name format: <alg>__<env_id>__<seed>__<timestamp>
  alg examples: ppo_relu_cchain, ppo_relu

Output CSVs:
- results/dmc_cchain_summary.csv (alg contains "cchain")
- results/dmc_ppo_summary.csv     (alg without "cchain")
Columns: env_id, succ_mean, succ_std, succ_mean_pm_std, ret_mean, ret_std, num_seeds
Note: DMC 没有 success 指标，这里用 Eval return 代替；success 列输出 NA。
"""
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

ENV_IDS = ["dog", "walker", "quadruped"]
SEEDS = [1, 2, 3, 4, 5]


def parse_run_name(run_name: str) -> Tuple[str, str, str]:
    parts = run_name.split("__")
    if len(parts) < 4:
        return None, None, None
    alg, env_id, seed = parts[0], parts[1], parts[2]
    return alg, env_id, seed


def load_last_eval(event_path: Path) -> float | None:
    ea = event_accumulator.EventAccumulator(str(event_path))
    ea.Reload()
    if "charts/Eval" not in ea.Tags().get("scalars", []):
        return None
    eval_vals = ea.Scalars("charts/Eval")
    if not eval_vals:
        return None
    return float(eval_vals[-1].value)


def summarize_group(runs_dir: Path, alg_filter, out_csv: Path) -> None:
    results: Dict[str, List[float]] = {env: [] for env in ENV_IDS}

    for event_file in runs_dir.rglob("events.out.tfevents.*"):
        run_name = event_file.parent.name
        alg, env_id, seed = parse_run_name(run_name)
        if alg is None or env_id not in ENV_IDS:
            continue
        if not alg_filter(alg):
            continue
        last_eval = load_last_eval(event_file)
        if last_eval is None:
            continue
        results[env_id].append(last_eval)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["task", "succ_mean", "succ_std", "succ_mean_pm_std", "ret_mean", "ret_std", "num_seeds"]
        )
        ret_means, ret_stds = [], []
        for env in ENV_IDS:
            vals = results[env]
            if not vals:
                writer.writerow([env, "NA", "NA", "NA", "NA", "NA", 0])
                continue
            arr = np.asarray(vals, dtype=np.float64)
            mean_ret = float(arr.mean())
            std_ret = float(arr.std(ddof=0))
            writer.writerow([env, "NA", "NA", "NA", mean_ret, std_ret, len(vals)])
            ret_means.append(mean_ret)
            ret_stds.append(std_ret)
        if ret_means:
            writer.writerow([])
            writer.writerow(
                [
                    "mean_across_tasks",
                    "NA",
                    "NA",
                    "NA",
                    float(np.mean(ret_means)),
                    float(np.mean(ret_stds)),
                    len(ret_means),
                ]
            )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    runs_dir = root / "runs"
    summarize_group(
        runs_dir,
        alg_filter=lambda a: "cchain" in a,
        out_csv=root / "results" / "dmc_cchain_summary.csv",
    )
    summarize_group(
        runs_dir,
        alg_filter=lambda a: "cchain" not in a,
        out_csv=root / "results" / "dmc_ppo_summary.csv",
    )


if __name__ == "__main__":
    main()
