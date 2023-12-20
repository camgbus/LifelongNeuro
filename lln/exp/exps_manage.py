"""Manage a "exps_summary.csv" file that summarizes the experiments going on in a given directory.
"""

import csv
import itertools
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_exps_summary(exps_path, splits, seeds):
    '''Creates a csv file with the summary of the experiments to be run.'''

    csv_file = os.path.join(exps_path, "exps_summary.csv")
    columns = ["exp_name", "split", "seed", "status", "hardware", "start", "end"]
    
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        
        exp_names = [item for item in os.listdir(exps_path) if os.path.isdir(os.path.join(exps_path, item))]
        
        for exp_name, split, seed in itertools.product(exp_names, splits, seeds):
            writer.writerow({"exp_name": exp_name, "split": split, "seed": seed, "status": "READY"})

def add_exps_to_summary(exps_path, exp_names, splits, seeds):
    '''Updates the csv file with new experiments to be run.'''
    csv_file = os.path.join(exps_path, "exps_summary.csv")
    columns = ["exp_name", "split", "seed", "status", "hardware", "start", "end"]
    
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        
        for exp_name, split, seed in itertools.product(exp_names, splits, seeds):
            writer.writerow({"exp_name": exp_name, "split": split, "seed": seed, "status": "READY"})

def visualize_ongoing_exps(exps_path, save=True):
    '''Visualizes the ongoing experiments in a given directory as a barplot.'''	
    csv_file = os.path.join(exps_path, "exps_summary.csv")
    df = pd.read_csv(csv_file)
    
    sns.set_style("whitegrid")  # Set the style to white grid
    
    # Specify the colors for the status
    status_colors = {"READY": "#d9d9d9", "RUNNING": "#a3cdf1", "FAILED": "#ff6b6b", "DONE": "#87c38f", "BLOCKED": "#ffe66d"}
    
    _, ax = plt.subplots(figsize=(12, 6))  # Specify the figure size
    
    ax = sns.countplot(data=df, x='exp_name', hue='status', palette=status_colors)
    
    # Specify the order for the legend
    legend_order = ["READY", "RUNNING", "FAILED", "DONE", "BLOCKED"]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[i] for i, label in enumerate(labels) if label in legend_order],
              [label for label in labels if label in legend_order],
              loc='upper right', bbox_to_anchor=(1.14, 1))
    
    ax.set_xlabel("Experiment")  # Change the x-axis label to "Experiment"
    ax.set_ylabel("Count")  # Change the y-axis label to "Count"
    
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
    
