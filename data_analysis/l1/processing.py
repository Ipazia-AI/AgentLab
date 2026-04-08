import gzip
import pickle
from pathlib import Path

import pandas as pd
from counting_functions import (
    calculate_verifications,
    count_action_report_frequencies,
    count_node_recoveries,
    count_pruned_nodes,
    count_retries,
)
from read_configuration import ExperimentConfig, load_task_categories_dataframe

from agentlab.analyze import inspect_results

TASK_NAME_PREFIX ="workarena.servicenow."
OFFSET = len(TASK_NAME_PREFIX)
TECHNIQUE = "hpa"
PKL_PREFIX = "step_"
PKL_SUFFIX = ".pkl.gz"
PKL_PATTERN = f"{PKL_PREFIX}*{PKL_SUFFIX}"

EXPERIMENT_CONFIGS_PATH = Path(__file__).resolve().parent / "l1_configs.yaml"
TASK_CATEGORIES = load_task_categories_dataframe(EXPERIMENT_CONFIGS_PATH)

def read_raw_data_and_reset_index(experiment_config: ExperimentConfig) -> pd.DataFrame:
    """
    Reads the raw data through AgentLab's default inspect_results method and resets the dataframe index.
    """
    df = inspect_results.load_result_df(experiment_config.exp_dir)
    df = df.reset_index()
    return df

def preprocess_raw_data(experiment_config: ExperimentConfig) -> pd.DataFrame:
    preprocessed_df = read_raw_data_and_reset_index(experiment_config)
    task_lookup = dict(zip(TASK_CATEGORIES["task_name"], TASK_CATEGORIES["category"]))

    preprocessed_df["env.task_name"] = [task_name[OFFSET:] for task_name in preprocessed_df["env.task_name"]]
    preprocessed_df["model"] = [experiment_config.id] * len(preprocessed_df)
    preprocessed_df["task_category"] = preprocessed_df["env.task_name"].map(lambda x: task_lookup.get(x, "Unknown"))
    new_cols = preprocessed_df["exp_dir"].apply(add_pkl_info_to_dataframe, is_genericagent=experiment_config.is_genericagent)
    preprocessed_df[["tree_evolution", "action_report"]] = new_cols
    preprocessed_df["action_report_frequencies"] = preprocessed_df.apply(count_action_report_frequencies, axis=1)

    if not experiment_config.is_genericagent:
        preprocessed_df["pruned_nodes"] = preprocessed_df["tree_evolution"].apply(count_pruned_nodes)
        preprocessed_df["task_retries"] = preprocessed_df["tree_evolution"].apply(count_retries)
        preprocessed_df["node_recoveries"] = preprocessed_df["tree_evolution"].apply(count_node_recoveries)
        preprocessed_df["action_verifications"] = preprocessed_df.apply(calculate_verifications, axis=1)
        # preprocessed_df["action_report_frequencies"] = preprocessed_df.apply(count_action_report_frequencies, axis=1)

    if experiment_config.save_csv:
        preprocessed_df.to_csv(f"preprocessed_l1_{experiment_config.id}.csv")
    
    return preprocessed_df

def add_pkl_info_to_dataframe(exp_dir: str | Path, is_genericagent: bool = False) -> pd.Series:
    exp_dir = Path(exp_dir)
    pkl_file_list = list(exp_dir.glob(PKL_PATTERN))
    pkl_file_list.sort(key=lambda x: int(Path(x.stem).stem.split("_")[-1]))
    tree_evolution = {}
    action_report = {}
    
    for index, pkl_file in enumerate(pkl_file_list):
        with gzip.open(pkl_file, "rb") as f:
            obj = pickle.load(f)
            if hasattr(obj.agent_info, "extra_info") and not is_genericagent:
                tree_evolution[f"{PKL_PREFIX}{index}"] = obj.agent_info.extra_info["hpa_plan"]
            action_report[f"{PKL_PREFIX}{index}"] = obj.obs["last_action_error"]
    
    if not tree_evolution:
        tree_evolution = None

    return pd.Series({
        "tree_evolution": tree_evolution, 
        "action_report": action_report
    })

def str_to_dict(s: str) -> dict:
    return eval(s)

def read_preprocessed_csv(path: Path) -> pd.DataFrame:
    preprocessed_df = pd.read_csv(path)
    preprocessed_df["tree_evolution"] = preprocessed_df["tree_evolution"].apply(str_to_dict)
    return preprocessed_df

