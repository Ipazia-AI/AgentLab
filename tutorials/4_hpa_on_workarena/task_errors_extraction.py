import argparse
from pathlib import Path

import pandas as pd

from agentlab.analyze import inspect_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract task errors from AgentLab results into a CSV file."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to the results folder (e.g. .../2026-02-17_23-39-12_hpaagent-...-on-workarena-l1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="tutorials/4_hpa_on_workarena/task_errors.csv",
        metavar="FILE",
        help="Output CSV file name (default: task_errors.csv)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        parser.error(f"Results path is not a directory: {results_dir}")

    result_batches = []
    for result_batch_dir in results_dir.iterdir():
        if result_batch_dir.name == ".DS_Store" or not result_batch_dir.is_dir():
            continue
        result_batches.append(inspect_results.load_result_df(result_batch_dir))

    result_df = pd.concat(result_batches)
    errors_df = result_df[~result_df["err_msg"].isna()].reset_index()
    errors_df = errors_df[["env.task_name", "env.task_seed", "err_msg", "exp_dir"]]
    errors_df.to_csv(args.output)


if __name__ == "__main__":
    main()

