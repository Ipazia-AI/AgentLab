from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass
class ExperimentConfig:
    id: str
    exp_dir: str | Path
    is_genericagent: bool
    n_steps: int
    save_csv: bool


def load_l1_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_task_categories_dataframe(config_path: Path) -> pd.DataFrame:
    config = load_l1_config(config_path)
    task_categories_path = Path(__file__).resolve().parent / config.get("task_categories")
    return pd.read_csv(task_categories_path)

def load_experiment_config_from_id(id: str, config_path: Path) -> ExperimentConfig:
    config = load_l1_config(config_path)
    for run in config.get("runs"):
        if run.get("id") == id:
            run["exp_dir"] = Path(run["exp_dir"]).expanduser().resolve()
            return ExperimentConfig(**run)
    raise ValueError(f"Experiment config with id {id} not found")

def load_experiment_configs_from_ids(ids: list[str], config_path: Path) -> list[ExperimentConfig]:
    return [load_experiment_config_from_id(id=id, config_path=config_path) for id in ids]