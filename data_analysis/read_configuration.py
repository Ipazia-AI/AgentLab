from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd
import yaml
from browsergym.workarena.tasks.compositional.utils.curriculum import AGENT_CURRICULUM

TASK_NAME_PREFIX ="workarena.servicenow."
L2_TASK_NAME_SUFFIX = "-l2"
EXPERIMENT_CONFIGS_PATH = Path(__file__).resolve().parent / "configs.yaml"

class Subset(StrEnum):
    L1 = "l1"
    L2 = "l2"

@dataclass
class ExperimentConfig:
    id: str
    subset: Subset
    exp_dir: str | Path
    is_genericagent: bool
    n_steps: int
    save_csv: bool

    def get_task_lookup(self) -> dict:
        if self.subset == Subset.L1:
            return dict(zip(TASK_CATEGORIES["task_name"], TASK_CATEGORIES["category"]))
        elif self.subset == Subset.L2:
            return {task.__name__.lower(): key for key in AGENT_CURRICULUM.keys() for bucket in AGENT_CURRICULUM[key]["buckets"] for task in bucket}

    def get_task_name_offsets(self) -> tuple[int, int] | tuple[int, None]:
        if self.subset == Subset.L1:
            return (len(TASK_NAME_PREFIX), None)
        elif self.subset == Subset.L2:
            return (len(TASK_NAME_PREFIX), len(L2_TASK_NAME_SUFFIX))
    
    def task_category_mapping(self, task_lookup: dict) -> callable:
        if self.subset == Subset.L1:
            return lambda task_name: task_lookup.get(task_name, "Unknown")
        elif self.subset == Subset.L2:
            return lambda task_name: task_lookup.get("".join(task_name.split("-")) + "task", "Unknown")


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_task_categories_dataframe(config_path: Path) -> pd.DataFrame:
    config = load_config(config_path)
    task_categories_path = Path(__file__).resolve().parent / config.get("task_categories_l1")
    return pd.read_csv(task_categories_path)

TASK_CATEGORIES = load_task_categories_dataframe(EXPERIMENT_CONFIGS_PATH)

def load_experiment_config_from_id(id: str, config_path: Path) -> ExperimentConfig:
    config = load_config(config_path)
    for run in config.get("runs"):
        if run.get("id") == id:
            run["exp_dir"] = Path(run["exp_dir"]).expanduser().resolve()
            return ExperimentConfig(**run)
    raise ValueError(f"Experiment config with id {id} not found")

def load_experiment_configs_from_ids(ids: list[str], config_path: Path) -> list[ExperimentConfig]:
    return [load_experiment_config_from_id(id=id, config_path=config_path) for id in ids]