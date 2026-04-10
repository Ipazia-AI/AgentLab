import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_combined_reward_step_count(df: pd.DataFrame, save: bool = False):
    # Set figsize height to accommodate thicker bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
    sorted_df = df.sort_values(by=["model", "cum_reward"], ascending=[False, True])

    # Plotting parameters: width = total space used by the group; gap = space between the bars within the group
    plot_params = {
        "data": sorted_df,
        "y": "env.task_name",
        "hue": "model",
        "orient": 'h',
        "errorbar": None,
        "width": 0.6,    
    }

    # Left Plot: Average Reward
    sns.barplot(ax=ax1, x="cum_reward", **plot_params)
    ax1.set_xlabel("Average Reward")
    ax1.set_ylabel("Task Name")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.6)

    # Right Plot: Average Step Count
    sns.barplot(ax=ax2, x="n_steps", **plot_params)
    ax2.set_xlabel("Average Step Count")
    ax2.set_ylabel("") 
    ax2.grid(True, axis="x", linestyle="--", alpha=0.6)
    ax2.axvline(x=15, linestyle="--", color="red", linewidth=1.5)
    ax2.set_xlim(0, 30)

    # Legend & Layout
    handles, labels = ax1.get_legend_handles_labels()
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    fig.legend(handles, labels, loc='upper center', title="Model", bbox_to_anchor=(0.5, 0.98), ncol=len(labels))

    # Space between the subplots
    plt.subplots_adjust(wspace=0.15) 
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save:
        models_str = '_'.join(df.model.unique())
        plt.savefig(f"plots/combined_metrics_{models_str}.png", bbox_inches="tight", dpi=200)
    
    plt.show()