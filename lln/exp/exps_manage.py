"""Manage a "exps_overview.csv" file that summarizes the experiments going on in a given directory.
"""

import csv
import itertools
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lln.utils.io import load_json

def create_exps_overview(exps_path, config_names, splits, seeds, add_training_with_all_data=False):
    '''Creates a csv file with the summary of the experiments to be run.
    
    Args:
        exps_path (str): path to the directory where the experiment directory will be created.
        config_names (list): list of config names.
        splits (list): list of split names.
        seeds (list): list of seeds.
    '''

    csv_file = os.path.join(exps_path, "exps_overview.csv")
    columns = ["exp", "config", "split", "seed", "state", "blocked_by", "hardware", "start", "end"]
    
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        
        exp_names = [item for item in os.listdir(exps_path) if os.path.isdir(os.path.join(exps_path, item))]
        
        if add_training_with_all_data:
            splits = ['ALL'] + list(splits)

        for exp_name, config_name, split, seed in itertools.product(exp_names, config_names, splits, seeds):
            writer.writerow({"exp": exp_name, "config": config_name, "split": str(split), "seed": seed, "state": "READY"})

def add_exps_to_summary(exps_path, exp_names, config_names, splits, seeds, add_training_with_all_data=False):
    '''Updates the csv file with new experiments to be run.
    
    Args:
        exps_path (str): path to the directory where the experiment directory will be created.
        exp_names (list): list of experiment names.
        config_names (list): list of config names.
        splits (list): list of split names.
        seeds (list): list of seeds.
    '''
    csv_file = os.path.join(exps_path, "exps_overview.csv")
    columns = ["exp", "config", "split", "seed", "state", "blocked_by", "hardware", "start", "end"]
    
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        
        if add_training_with_all_data:
            splits = ['ALL'] + list(splits)

        for exp_name, config_name, split, seed in itertools.product(exp_names, config_names, splits, seeds):
            
            config = load_json(path=os.path.join(exps_path, exp_name), file_name=f'config_{config_name}')
            blocked_by = config.get('needs', [])
            # Remove the needed_config items that fulfill the condition
            blocked_by = [needed_config for needed_config in blocked_by if 
                           not any(row['exp'] == exp_name and row['config'] == needed_config and 
                                   row['split'] == split and row['seed'] == seed and row['state'] == 'DONE' 
                                   for row in csv.DictReader(open(csv_file, encoding='utf-8')))]
            state = "READY" if len(blocked_by) == 0 else "BLOCKED"

            writer.writerow({"exp": exp_name, "config": config_name, "split": str(split), "seed": 
                seed, "state": state, "blocked_by": '-'.join(blocked_by)})

def visualize_ongoing_exps(exps_path, save=True):
    '''Visualizes the ongoing experiments in a given directory as a barplot.
    
    Args:
        exps_path (str): path to the directory of the experiment.
        save (bool): whether to save the plot or not.
    '''	
    csv_file = os.path.join(exps_path, "exps_overview.csv")
    df = pd.read_csv(csv_file)
    
    sns.set_style("whitegrid")  # Set the style to white grid
    
    # Specify the colors for the state
    state_colors = {"READY": "#d9d9d9", "RUNNING": "#a3cdf1", "FAILED": "#ff6b6b", "DONE": "#87c38f", "BLOCKED": "#ffe66d"}
    
    _, ax = plt.subplots(figsize=(12, 6))  # Specify the figure size
    
    df['exp_config'] = df['exp'] + ' | ' + df['config']
    ax = sns.countplot(data=df, x='exp_config', hue='state', palette=state_colors)

    # Specify the order for the legend
    legend_order = ["READY", "RUNNING", "FAILED", "DONE", "BLOCKED"]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[i] for i, label in enumerate(labels) if label in legend_order],
              [label for label in labels if label in legend_order],
              loc='upper right', bbox_to_anchor=(1.14, 1))

    ax.set_xlabel("Experiment")  # Change the x-axis label to "Experiment"
    ax.set_ylabel("Count")  # Change the y-axis label to "Count"
    
    plt.xticks(rotation=90, ha='right', wrap=False)

    # Add count labels on top of each bar if count is not 0
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

    plt.title(f"Ongoing experiments in {os.path.basename(exps_path)}")

    if save:
        output_file = os.path.join(exps_path, "ongoing_exps.svg")
        plt.savefig(output_file, bbox_inches='tight')
    else:
        plt.show()
    
