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
from read_configuration import ExperimentConfig

from agentlab.analyze import inspect_results

PKL_PREFIX = "step_"
PKL_SUFFIX = ".pkl.gz"
PKL_PATTERN = f"{PKL_PREFIX}*{PKL_SUFFIX}"

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"

def read_raw_data_and_reset_index(experiment_config: ExperimentConfig) -> pd.DataFrame:
    """
    Reads the raw data through AgentLab's default inspect_results method and resets the dataframe index.
    """
    df = inspect_results.load_result_df(experiment_config.exp_dir)
    df = df.reset_index()
    return df

def preprocess_raw_data(experiment_config: ExperimentConfig) -> pd.DataFrame:
    preprocessed_df = read_raw_data_and_reset_index(experiment_config)
    task_lookup = experiment_config.get_task_lookup()
    prefix_offset, suffix_offset = experiment_config.get_task_name_offsets()

    preprocessed_df["env.task_name"] = [task_name[prefix_offset:-suffix_offset] if suffix_offset else task_name[prefix_offset:] for task_name in preprocessed_df["env.task_name"]]
    preprocessed_df["model"] = [experiment_config.id] * len(preprocessed_df)
    preprocessed_df["task_category"] = preprocessed_df["env.task_name"].map(experiment_config.task_category_mapping(task_lookup))
    new_cols = preprocessed_df["exp_dir"].apply(add_pkl_info_to_dataframe, is_genericagent=experiment_config.is_genericagent)
    preprocessed_df[["tree_evolution", "action_report", "actions", "axtree_objects"]] = new_cols
    preprocessed_df["action_report_frequencies"] = preprocessed_df.apply(count_action_report_frequencies, axis=1)

    if not experiment_config.is_genericagent:
        preprocessed_df["pruned_nodes"] = preprocessed_df["tree_evolution"].apply(count_pruned_nodes)
        preprocessed_df["task_retries"] = preprocessed_df["tree_evolution"].apply(count_retries)
        preprocessed_df["node_recoveries"] = preprocessed_df["tree_evolution"].apply(count_node_recoveries)
        preprocessed_df["action_verifications"] = preprocessed_df.apply(calculate_verifications, axis=1)

    if experiment_config.save_csv:
        DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        preprocessed_df.to_csv(DATA_DIRECTORY / f"preprocessed_{experiment_config.subset}_{experiment_config.id}.csv", index=False)
    
    return preprocessed_df

def add_pkl_info_to_dataframe(exp_dir: str | Path, is_genericagent: bool = False) -> pd.Series:
    exp_dir = Path(exp_dir)
    pkl_file_list = list(exp_dir.glob(PKL_PATTERN))
    pkl_file_list.sort(key=lambda x: int(Path(x.stem).stem.split("_")[-1]))
    tree_evolution = {}
    action_report = {}
    actions = {}
    axtree_objects = {}
    
    for index, pkl_file in enumerate(pkl_file_list):
        with gzip.open(pkl_file, "rb") as f:
            obj = pickle.load(f)
            if hasattr(obj.agent_info, "extra_info") and not is_genericagent:
                tree_evolution[f"{PKL_PREFIX}{index}"] = obj.agent_info.extra_info["hpa_plan"]
            action_report[f"{PKL_PREFIX}{index}"] = (obj.obs or {}).get("last_action_error") # Handle cases where obj.obs is None
            actions[f"{PKL_PREFIX}{index}"] = obj.action
            axtree_objects[f"{PKL_PREFIX}{index}"] = (obj.obs or {}).get("axtree_txt", "")

    return pd.Series({
        "tree_evolution": tree_evolution, 
        "action_report": action_report,
        "actions": actions,
        "axtree_objects": axtree_objects
    })

def str_to_dict(s: str) -> dict:
    return eval(s)

def read_preprocessed_csv(path: Path) -> pd.DataFrame:
    preprocessed_df = pd.read_csv(path)
    preprocessed_df["tree_evolution"] = preprocessed_df["tree_evolution"].apply(str_to_dict)
    preprocessed_df["action_report"] = preprocessed_df["action_report"].apply(str_to_dict)
    preprocessed_df["action_report_frequencies"] = preprocessed_df["action_report_frequencies"].apply(str_to_dict)
    preprocessed_df["actions"] = preprocessed_df["actions"].apply(str_to_dict)
    preprocessed_df["axtree_objects"] = preprocessed_df["axtree_objects"].apply(str_to_dict)
    if "action_verifications" in preprocessed_df.columns.values:
        preprocessed_df["action_verifications"] = preprocessed_df["action_verifications"].apply(str_to_dict)
    return preprocessed_df