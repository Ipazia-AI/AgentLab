import pandas as pd
import typer
from processing import DATA_DIRECTORY, preprocess_raw_data, read_preprocessed_csv
from read_configuration import (
    EXPERIMENT_CONFIGS_PATH,
    load_experiment_configs_from_ids,
)

app = typer.Typer(no_args_is_help=True)

@app.command()
def process_data(ids: list[str] = typer.Argument(..., help="The IDs of the experiments to process. If not provided, an error will be raised.")) -> pd.DataFrame:
    preprocessed_df_list = []
    experiment_configs = load_experiment_configs_from_ids(ids=ids, config_path=EXPERIMENT_CONFIGS_PATH)

    for experiment_config in experiment_configs:
        if (DATA_DIRECTORY / f"preprocessed_{experiment_config.subset}_{experiment_config.id}.csv").exists():
            preprocessed_df_list.append(read_preprocessed_csv(path=DATA_DIRECTORY / f"preprocessed_{experiment_config.subset}_{experiment_config.id}.csv"))
        else:
            preprocessed_df_list.append(preprocess_raw_data(experiment_config=experiment_config))
    
    preprocessed_df = pd.concat(preprocessed_df_list).reset_index(drop=True)
    
    return preprocessed_df

if __name__ == "__main__":
    app()