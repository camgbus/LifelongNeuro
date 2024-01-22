"""Evaluate the results from each experiment run, then calculate merge these into summarized results.
"""
import os
import itertools
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
from lln.plotting.seaborn.plots import highlight_legend_titles
    
def average_runs(exps_path, exp_names, k_folds=None, seeds=[0], run_names=None):
    '''Summarizes the results of multiple experiment runs, e.g. splits or seeds. Each run should 
    have a subdirectory with the corresponding name inside the experiment directory.'''
    if run_names is None:
        run_names = [f'SPLIT_{split}_SEED_{seed}' for split, seed in itertools.product(range(k_folds), seeds)]
    else:
        assert k_folds is None and seeds is [0]
    for exp_name in exp_names:
        exp_path = os.path.join(exps_path, exp_name)
        
        # Average the results df
        results_df = None
        for run_name in run_names:
            run_path = os.path.join(exp_path, run_name)
            assert os.path.exists(run_path), f"Experiment {exp_name}, run {run_name} not found"
            df = pd.read_csv(os.path.join(run_path, 'results.csv'))
            df.insert(0, 'Run', run_name)
            results_df = df if results_df is None else pd.concat([results_df, df], ignore_index=True)
        results_df = _set_rows_mean_std(results_df)
        results_df.to_csv(os.path.join(exp_path, 'results.csv'), index=False)
        
        # Average the loss_trajectory and progress files
        for file_name in ['loss_trajectory', 'progress']:
            progress_df = None
            for run_name in run_names:
                run_path = os.path.join(exp_path, run_name)
                df = pd.read_csv(os.path.join(run_path, 'trainer', file_name+'.csv'))
                df.insert(0, 'Run', run_name)
                progress_df = df if progress_df is None else pd.concat([progress_df, df], ignore_index=True)
            progress_df = _set_rows_mean_std(progress_df, non_avg=['Dataset', 'Epoch'])
            progress_df.to_csv(os.path.join(exp_path, file_name+'.csv'), index=False)
        
def _set_rows_mean_std(df, non_avg = ['Dataset']):
    non_avg_values = [[(x, col) for x in df[col].unique()] for col in non_avg]
    for values in itertools.product(*non_avg_values):
        # Filter the df to only contain the rows with the given values
        filter_condition = True
        for value, col in values:
            filter_condition &= (df[col] == value)
        filtered_df = df[filter_condition]
        # Calculate the mean and std of the filtered df and add them to the df
        if not filtered_df.empty:
            mean_values = filtered_df.drop(non_avg + ['Run'], axis=1).mean().to_dict()
            mean_values['Run'] = 'Mean'
            std_values = filtered_df.drop(non_avg + ['Run'], axis=1).std().to_dict()
            std_values['Run'] = 'Std'
            for value, col in values:
                mean_values[col] = value
                std_values[col] = value
            df = pd.concat([df, pd.DataFrame([mean_values]), pd.DataFrame([std_values])], ignore_index=True)
    return df

import matplotlib.font_manager as fm

def plot_progress(exps_path, exp_names, save=True):
    '''Plots the progress of multiple runs and experiments in one plot. Runs from the same 
    experiment have the same color.'''
    plotting_df = None
    for exp_name in exp_names:
        exp_path = os.path.join(exps_path, exp_name)
        progress_df = pd.read_csv(os.path.join(exp_path, 'progress.csv'))
        loss_trajectory_df = pd.read_csv(os.path.join(exp_path, 'loss_trajectory.csv'))
        combined_df = pd.merge(progress_df, loss_trajectory_df, on=['Epoch', 'Dataset', 'Run'])
        combined_df.insert(0, 'Method', exp_name)
        plotting_df = combined_df if plotting_df is None else pd.concat([plotting_df, combined_df], ignore_index=True)
    
    # Filter out rows with Run == "Mean" or "Std"
    plotting_df = plotting_df[~plotting_df['Run'].isin(['Mean', 'Std'])]
    
    # Create lineplots
    for metric in plotting_df.columns:
        if metric not in ['Epoch', 'Dataset', 'Method', 'Run']:
            _, ax = plt.subplots(figsize=(8, 6))  # Specify the figure size
            ax = sns.lineplot(data=plotting_df, x='Epoch', y=metric, hue='Method', style='Dataset')
            ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1))
            highlight_legend_titles(ax, legend_titles = ['Method', 'Dataset'])
            
            # Save plots
            if save:
                output_file = os.path.join(exps_path, f"exps_trajectories_{metric}.svg")
                plt.savefig(output_file, bbox_inches='tight')
            else:
                plt.show()
            plt.close()