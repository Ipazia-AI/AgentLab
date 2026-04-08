from pathlib import Path

import pandas as pd
import typer
from processing import preprocess_raw_data, read_preprocessed_csv
from read_configuration import (
    load_experiment_configs_from_ids,
)

EXPERIMENT_CONFIGS_PATH = Path(__file__).resolve().parent / "l1_configs.yaml"

app = typer.Typer(no_args_is_help=True)

@app.command()
def process_l1_data(ids: list[str] = typer.Argument(..., help="The IDs of the experiments to process. If not provided, an error will be raised."),
    from_csv: bool = typer.Option(default=False, help="Whether a CSV file of the preprocessed data already exists. If True, the function will read the CSV file instead of processing the raw data."),
) -> pd.DataFrame:
    if from_csv:
        preprocessed_df_list = [read_preprocessed_csv(path=Path(f"preprocessed_l1_{id}.csv")) for id in ids]
    else:
        experiment_configs = load_experiment_configs_from_ids(ids=ids, config_path=EXPERIMENT_CONFIGS_PATH)
        preprocessed_df_list = [preprocess_raw_data(experiment_config=experiment_config) for experiment_config in experiment_configs]
    preprocessed_df = pd.concat(preprocessed_df_list).reset_index(drop=True)
    return preprocessed_df

if __name__ == "__main__":
    app()