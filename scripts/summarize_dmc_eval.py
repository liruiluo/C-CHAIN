#!/usr/bin/env python
"""
Summarize Eval and Eval_Horizon from TensorBoard logs of DMC runs.

Outputs a CSV with columns:
run_name, env_id, seed, alg, eval_step, eval_return, eval_horizon
"""
import csv
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def parse_run_name(run_name: str):
    # Expected pattern: <alg>__<env_id>__<seed>__<timestamp>
    parts = run_name.split("__")
    if len(parts) < 4:
        return None, None, None
    alg, env_id, seed = parts[0], parts[1], parts[2]
    return alg, env_id, seed


def summarize_run(events_path: Path):
    ea = event_accumulator.EventAccumulator(str(events_path))
    ea.Reload()
    eval_tags = [tag for tag in ea.Tags().get("scalars", []) if tag in ("charts/Eval", "charts/Eval_Horizon")]
    if not eval_tags:
        return None
    eval_vals = ea.Scalars("charts/Eval") if "charts/Eval" in eval_tags else []
    horizon_vals = ea.Scalars("charts/Eval_Horizon") if "charts/Eval_Horizon" in eval_tags else []
    if not eval_vals:
        return None
    # Take the last eval entry
    last_eval = eval_vals[-1]
    last_horizon = horizon_vals[-1].value if horizon_vals else None
    return {
        "eval_step": int(last_eval.step),
        "eval_return": float(last_eval.value),
        "eval_horizon": None if last_horizon is None else float(last_horizon),
    }


def main():
    root = Path(__file__).resolve().parent.parent
    runs_dir = root / "runs"
    out_csv = root / "results" / "dmc_eval_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for event_file in runs_dir.rglob("events.out.tfevents.*"):
        run_name = event_file.parent.name
        alg, env_id, seed = parse_run_name(run_name)
        if alg is None:
            continue
        summary = summarize_run(event_file)
        if summary is None:
            continue
        rows.append(
            {
                "run_name": run_name,
                "alg": alg,
                "env_id": env_id,
                "seed": seed,
                "eval_step": summary["eval_step"],
                "eval_return": summary["eval_return"],
                "eval_horizon": summary["eval_horizon"],
            }
        )

    if not rows:
        print("No eval data found in runs/")
        sys.exit(0)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_name", "alg", "env_id", "seed", "eval_step", "eval_return", "eval_horizon"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
