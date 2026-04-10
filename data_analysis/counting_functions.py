from collections import Counter

import pandas as pd

ACTION_VERIFICATION_SCHEMA = [
    "action_error_success", 
    "action_error_failure", 
    "verification_function_success", 
    "verification_function_failure", 
    "no_verification", 
    "first_step"
]

def count_pruned_nodes(tree_evolution: dict) -> int:
    return sum(len(step.get("tree_update", {}).get("prune", [])) for step in tree_evolution.values())

def count_action_report_frequencies(row: pd.Series) -> dict:
    error_count = 0
    success_count = 0
    n_steps = row["n_steps"] + 1 # step 0 is not counted in n_steps column but is counted in action_report
    for _, error_report in row['action_report'].items():
        if error_report != "":
            error_count += 1
        else:
            success_count += 1
    divisor = n_steps if n_steps > 0 else 1
    return {"error_frequency": error_count / divisor, "success_frequency": success_count / divisor}

def count_retries(tree_evolution: dict) -> int:
    """
    Computes the maximum retry value encountered during the tree evolution.
    """
    max_retry = 0
    
    for step in tree_evolution.values():
        # Access the list of expansion attempts in this step
        expansions = step.get("expansions", [])
        
        for entry in expansions:
            # Update max_retry if the current 'retry' value is higher
            current_retry = entry.get("retry", 0)
            if current_retry > max_retry:
                max_retry = current_retry
                
    return max_retry
    
def count_node_recoveries(tree_evolution: dict) -> int:
    return sum(len(step.get("recoveries", [])) for step in tree_evolution.values())

def calculate_verifications(row: pd.Series) -> dict:
    # Initialize Counter for automatic key handling
    counts = Counter()
    tree = row["tree_evolution"]
    n_steps = row["n_steps"]
    
    # Iterate through step in the tree
    for step_index, step_data in enumerate(tree.values()):
        v = step_data.get("action_verification")
        
        if isinstance(v, dict):
            source = v.get("verification_source")
            success = v.get("is_success")
            
            if source == "no_verification":
                counts["no_verification"] += 1
            else:
                # Dynamic key: "action_error_success", "verification_function_failure", etc.
                key = f"{source}_{'success' if success else 'failure'}"
                counts[key] += 1
        elif step_index == 0:
            # action verifications at step 0 are None
            counts["first_step"] += 1
        else:
            counts["action_error_failure"] += 1
    
    # Return normalized dictionary (handling division by zero)
    divisor = n_steps if n_steps > 0 else 1
    return {k: counts[k] / divisor for k in ACTION_VERIFICATION_SCHEMA}