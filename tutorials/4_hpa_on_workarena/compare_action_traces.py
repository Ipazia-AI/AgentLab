import argparse
from pathlib import Path

import pandas as pd

from agentlab.experiments.loop import yield_all_exp_results


def load_action_trace_df(study_dir: Path) -> pd.DataFrame:
    rows = []
    for exp_result in yield_all_exp_results(study_dir, progress_fn=None):
        exp_args = exp_result.exp_args
        task_name = exp_args.env_args.task_name
        task_seed = exp_args.env_args.task_seed
        agent_name = exp_args.agent_args.agent_name

        steps_info = exp_result.steps_info
        for i, step_info in enumerate(steps_info):
            if step_info.action is None:
                continue

            obs = step_info.obs if isinstance(step_info.obs, dict) else {}
            agent_info = step_info.agent_info if isinstance(step_info.agent_info, dict) else {}
            extra_info = agent_info.get("extra_info", {})
            if not isinstance(extra_info, dict):
                extra_info = {"value": repr(extra_info)}

            next_error = None
            if i + 1 < len(steps_info):
                next_obs = steps_info[i + 1].obs
                if isinstance(next_obs, dict):
                    next_error = next_obs.get("last_action_error")

            rows.append(
                {
                    "task_name": task_name,
                    "task_seed": task_seed,
                    "agent_name": agent_name,
                    "step": step_info.step,
                    "action": step_info.action,
                    "reward": step_info.reward,
                    "raw_reward": step_info.raw_reward,
                    "url": obs.get("url"),
                    "last_action_error_before": obs.get("last_action_error"),
                    "last_action_error_after": next_error,
                    "think": agent_info.get("think"),
                    "oracle_trace": extra_info.get("oracle_trace"),
                    "extra_info": extra_info,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare candidate actions against oracle traces.")
    parser.add_argument("--oracle-study", type=Path, required=True)
    parser.add_argument("--candidate-study", type=Path, required=True)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("action_trace_comparison.csv"),
        help="Where to write the per-step comparison table.",
    )
    args = parser.parse_args()

    oracle_df = load_action_trace_df(args.oracle_study)
    candidate_df = load_action_trace_df(args.candidate_study)

    if oracle_df.empty:
        raise ValueError(f"No action trace found in oracle study: {args.oracle_study}")
    if candidate_df.empty:
        raise ValueError(f"No action trace found in candidate study: {args.candidate_study}")

    join_keys = ["task_name", "task_seed", "step"]
    merged = oracle_df.merge(
        candidate_df,
        on=join_keys,
        suffixes=("_oracle", "_candidate"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("No aligned rows found on (task_name, task_seed, step).")

    merged["action_match"] = merged["action_oracle"] == merged["action_candidate"]

    summary = (
        merged.groupby(["task_name", "task_seed"], as_index=False)["action_match"]
        .mean()
        .rename(columns={"action_match": "step_action_match_rate"})
    )
    overall = merged["action_match"].mean()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    summary.to_csv(args.out_csv.with_name(args.out_csv.stem + "_summary.csv"), index=False)

    print(f"Compared steps: {len(merged)}")
    print(f"Overall exact action match: {overall:.3f}")
    print(f"Per-step table: {args.out_csv}")
    print(f"Per-task summary: {args.out_csv.with_name(args.out_csv.stem + '_summary.csv')}")


if __name__ == "__main__":
    main()
