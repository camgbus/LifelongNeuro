"""Evaluate the results from each experiment run, then calculate merge these into summarized results.
"""
import os
import json
import itertools
import pandas as pd
import seaborn as sns
import pickle
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
from lln.plotting.seaborn.plots import highlight_legend_titles
import matplotlib.font_manager as fm
from lln.exp.formatted_summary import shape_format_summarized_df
from lln.exp.results_plotting import per_experiment_confusion_matrix, per_experiment_feature_attributions, per_experiment_per_timepoint_confusion_matrix
import sys

class ExperimentSummerizer:
    '''A class that summarizes and extracts certain results from an experiment
    '''
    def __init__(self, exps_path, exp_name, run_names=None, config_name='train'):
        self.exps_path = exps_path
        self.exp_name = exp_name
        self.exp_path = os.path.join(self.exps_path, exp_name)
        self.config = json.load(open(os.path.join(self.exp_path, f"config_{config_name}.json"), 'r'))
        self.config_suffix = f"_{config_name}"
        # Determine the run_names that summarize_run will run on
        if run_names is None:
            run_names = [f'SPLIT_{split}_SEED_{seed}{self.config_suffix}' for split, seed in itertools.product(range(self.config['nr_splits']), self.config['seeds'])]
        self.run_names = run_names
    
    def summarize_average_runs(self, epoch=None, state_selection_dataset='Val', state_selection_metric='B-Acc.', higher_is_better=True):
        '''Summarizes the results of multiple experiment runs, and averages them.
        If epochs is None, the best epoch is chosen for each fold on the state_selection_metric and state_selection_dataset.
        If it is an array of epochs, the results for each epoch are summarized.
        '''
        for run_name in self.run_names:
            print(f"Summarizing run {run_name}")
            self.summarize_run(run_name, epoch, state_selection_dataset, state_selection_metric, higher_is_better) 
        self.average_runs()
        
    def plot_per_experiment_results(self, labels, subject_splits = ['Train', 'Val', 'Test'], feature_attributions=False):
        '''Plot the results of the experiments as confusion matrices.'''
        exps_df = merge_in_df(self.exps_path, [self.exp_name], subject_splits=subject_splits, config_suffix=self.config_suffix)
        exps_dfs = {split_name: exps_df[exps_df['split'] == split_name] for split_name in subject_splits}
        for split_name in subject_splits:
            for weighted in [True, False]:
                per_experiment_per_timepoint_confusion_matrix(exps_dfs[split_name], self.exps_path, exp_name=self.exp_name, exp_better_name=self.exp_name, labels=labels, seeds=[0], weighted=weighted, split_name=split_name)
                per_experiment_confusion_matrix(exps_dfs[split_name], self.exps_path, exp_name=self.exp_name, exp_better_name=self.exp_name, labels=labels, seeds=[0], weighted=weighted, split_name=split_name)
            if False: #feature_attributions:
                for only_matching in [None, 0, 1]:
                    per_experiment_feature_attributions(exps_dfs[split_name], self.exps_path, exp_name=self.exp_name, exp_better_name=self.exp_name, labels=labels, seeds=[0], only_matching=only_matching, split_name=split_name)
    
    def summarize_run(self, run_name, epoch=None, state_selection_dataset='Val', state_selection_metric='B-Acc.', higher_is_better=True):
        '''Export a results.csv file for each run containing the best epoch for each metric and dataset.
        For each epoch, export as well results_<epoch>.csv files containing the results for each epoch.
        '''
        run_path = os.path.join(self.exp_path, run_name)
        # Looks for the best epoch for a given metric and dataset
        int_results_path = os.path.join(run_path, 'trainer')

        df_progess = pd.read_csv(os.path.join(int_results_path, "progress.csv"))
        df_loss_trajectory = pd.read_csv(os.path.join(int_results_path, "loss_trajectory.csv"))

        if epoch is None:
            if 'Loss' in state_selection_metric:
                df = df_loss_trajectory
            else:
                df = df_progess
                
            df_selection = df[df['Dataset'] == state_selection_dataset].reset_index(drop=True)
            if higher_is_better:
                best_epoch = df_selection['Epoch'].iloc[df_selection[state_selection_metric].idxmax()]
            else:
                best_epoch = df_selection['Epoch'].iloc[df_selection[state_selection_metric].idxmin()]
        else:
            best_epoch = epoch

        df_progess = df_progess[df_progess['Epoch'] == best_epoch]
        df_loss_trajectory = df_loss_trajectory[df_loss_trajectory['Epoch'] == best_epoch]

        df = pd.merge(df_progess, df_loss_trajectory, on=['Epoch', 'Dataset'], how='inner')

        results_path = os.path.join(run_path, f'results.csv')
        df.to_csv(results_path, index=False)
    
    def average_runs(self):
        '''Summarizes the results of multiple experiment runs, e.g. splits or seeds. Each run should 
        have a subdirectory with the corresponding name inside the experiment directory.'''
        if not os.path.exists(os.path.join(self.exp_path, f'results.csv')):
            results_df = None
            for run_name in self.run_names:
                run_path = os.path.join(self.exp_path, run_name)
                assert os.path.exists(run_path), f"Experiment {self.exp_name}, run {run_name} not found"
                df = pd.read_csv(os.path.join(run_path, f'results.csv'))
                df.insert(0, 'Run', run_name)
                results_df = df if results_df is None else pd.concat([results_df, df], ignore_index=True)
            results_df = self._set_rows_mean_std(results_df)
            results_df.to_csv(os.path.join(self.exp_path, f'results{self.config_suffix}.csv'), index=False)
                
            # Average the loss_trajectory and progress files
            if 'train' in self.config_suffix:
                for file_name in ['loss_trajectory', 'progress']:
                    progress_df = None
                    for run_name in self.run_names:
                        run_path = os.path.join(self.exp_path, run_name)
                        df = pd.read_csv(os.path.join(run_path, 'trainer', file_name+'.csv'))
                        df.insert(0, 'Run', run_name)
                        progress_df = df if progress_df is None else pd.concat([progress_df, df], ignore_index=True)
                    progress_df = self._set_rows_mean_std(progress_df, non_avg=['Dataset', 'Epoch'])
                    progress_df.to_csv(os.path.join(self.exp_path, file_name+'.csv'), index=False)

    def _set_rows_mean_std(self, df, non_avg = ['Dataset']):
        non_avg_values = [[(x, col) for x in df[col].unique()] for col in non_avg]
        for values in itertools.product(*non_avg_values):
            # Filter the df to only contain the rows with the given values
            filter_condition = True
            for value, col in values:
                filter_condition &= (df[col] == value)
            filtered_df = df[filter_condition]
            # Calculate the mean and std of the filtered df and add them to the df
            if not filtered_df.empty:
                #mean_values = filtered_df.drop(non_avg + ['Run'], axis=1).mean().to_dict()
                mean_values = filtered_df.drop(non_avg + ['Run'], axis=1).apply(lambda x: x.iloc[0] if x.dtype != 'float64' else x.mean()).to_dict()
                mean_values['Run'] = 'Mean'
                #std_values = filtered_df.drop(non_avg + ['Run'], axis=1).std().to_dict()
                std_values = filtered_df.drop(non_avg + ['Run'], axis=1).apply(lambda x: x.iloc[0] if x.dtype != 'float64' else x.std()).to_dict()
                std_values['Run'] = 'Std'
                for value, col in values:
                    mean_values[col] = value
                    std_values[col] = value
                df = pd.concat([df, pd.DataFrame([mean_values]), pd.DataFrame([std_values])], ignore_index=True)
        return df
    
    def select_best_epoch_global(self, state_selection_dataset='Val', state_selection_metric='B-Acc.', higher_is_better=True):
        dfs_selection = []
        for run_name in self.run_names:
            run_path = os.path.join(self.exp_path, run_name)
            if 'Loss' in state_selection_metric:
                dfs_selection.append(pd.read_csv(os.path.join(run_path, 'trainer', "loss_trajectory.csv")))
            else:
                dfs_selection.append(pd.read_csv(os.path.join(run_path, 'trainer', "progress.csv")))
        df_selection = pd.concat(dfs_selection, ignore_index=True)
        df_selection = df_selection.groupby(['Epoch', 'Dataset']).mean().reset_index()
        df_selection = df_selection[df_selection['Dataset'] == state_selection_dataset].reset_index(drop=True)
        if higher_is_better:
            best_epoch = df_selection['Epoch'].iloc[df_selection[state_selection_metric].idxmax()]
        else:
            best_epoch = df_selection['Epoch'].iloc[df_selection[state_selection_metric].idxmin()]
        return best_epoch

class ExperimentsSummerizer:
    '''A class that summarizes the results of multiple experiments, plotting the training 
    trajectories jointly and summarizing the result columns from results.csv into a joint file.
    '''
    def __init__(self, name, exps_path, exp_names=None, exp_better_names=None):
        if exp_names is None:
            exps_overview = pd.read_csv(os.path.join(exps_path, 'exps_overview.csv'))
            exp_names = exps_overview[exps_overview['state'] == 'DONE']['exp'].unique()
        self.exps_path = exps_path
        self.summary_path = os.path.join(exps_path, f'GROUP_SUMMARY_{name}')
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        self.exp_names = exp_names
        self.exp_better_names = exp_better_names
        
    def summarize_exps(self, datasets = ["Val", "Test"], metrics = ['B-Acc.', 'F1_mi'], higher_is_better = {'B-Acc.': True, 'F1_mi': True}, config=  'train'):
        '''Summarizes the results of multiple experiments, e.g. different hyperparameter settings.'''
        print(f"Summarizing the results from {len(self.exp_names)} experiments")
        exps_results = []
        for exp_name in self.exp_names:
            exp_results = pd.read_csv(os.path.join(self.exps_path, exp_name, f'results_{config}.csv'))
            exp_results = exp_results[(exp_results['Run'] == 'Mean') | (exp_results['Run'] == 'Std')]
            exp_name = exp_name if self.exp_better_names is None else self.exp_better_names[exp_name]
            exp_results.insert(0, 'Exp', exp_name)
            exps_results.append(exp_results)
        merged_results = pd.concat(exps_results, ignore_index=True)
        merged_results.to_csv(os.path.join(self.summary_path, 'exps_results.csv'), index=False)
        shape_format_summarized_df(path=self.summary_path, orig_file_name="exps_results", new_file_name="summ_exps_results", datasets=datasets, metrics=metrics, higher_is_better=higher_is_better)
            
    def plot_progress(self, save=True, fig_size=(18, 6)):
        '''Plots the progress of multiple runs and experiments in one plot. Runs from the same 
        experiment have the same color.'''
        plotting_df = None
        for exp_name in self.exp_names:
            exp_path = os.path.join(self.exps_path, exp_name)
            progress_df = pd.read_csv(os.path.join(exp_path, 'progress.csv'))
            loss_trajectory_df = pd.read_csv(os.path.join(exp_path, 'loss_trajectory.csv'))
            combined_df = pd.merge(progress_df, loss_trajectory_df, on=['Epoch', 'Dataset', 'Run'])
            exp_name = exp_name if self.exp_better_names is None else self.exp_better_names[exp_name]
            combined_df.insert(0, 'Method', exp_name)
            plotting_df = combined_df if plotting_df is None else pd.concat([plotting_df, combined_df], ignore_index=True)
        
        # Filter out rows with Run == "Mean" or "Std"
        plotting_df = plotting_df[~plotting_df['Run'].isin(['Mean', 'Std'])]
        
        # Create lineplots
        for metric in plotting_df.columns:
            if metric not in ['Epoch', 'Dataset', 'Method', 'Run']:
                
                line_styles = {'Train': (1, 1), 'Val': (2, 2), 'Test': (1, 0)}
                sns_plot = sns.lineplot(data=plotting_df, x='Epoch', y=metric, hue='Method', style='Dataset', dashes=line_styles)
                ax = plt.gca()
                
                # Place the legend outside the top right corner of the plot
                handles, labels = ax.get_legend_handles_labels()  # Get handles and labels from one of the subplots
                labels = ['\n' + label if label in ['Dataset'] else label for label in labels]
                ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1))

                # Adjust figure size and layout
                plt.gcf().set_size_inches(fig_size[0], fig_size[1])  # Set the figure size (width, height in inches)
                plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the padding of the figure. rect=[left, bottom, right, top] in normalized figure coordinates.

                # Save plots
                if save:
                    output_file = os.path.join(self.summary_path, f"exps_trajectories_{metric}.svg")
                    plt.savefig(output_file, bbox_inches='tight')
                else:
                    plt.show()
                plt.close()
                
def merge_in_df(exps_path, exp_names, exp_better_names = dict(), subject_splits = ['Val', 'Test'], config_suffix = "_train"):
    data = []
    for exp_name in exp_names:
        exp_path = os.path.join(exps_path, exp_name)
        config = json.load(open(os.path.join(exp_path, "config_train.json")))
        for split in range(config['nr_splits']):
            for seed in config['seeds']:
                run_name = f"SPLIT_{split}_SEED_{seed}{config_suffix}"
                run_splits = json.load(open(os.path.join(exp_path, run_name, "split.json"), 'rb'))
                targets = pickle.load(open(os.path.join(exp_path, run_name, "model_outputs", "targets.pkl"), 'rb'))
                predictions = pickle.load(open(os.path.join(exp_path, run_name, "model_outputs", "predictions.pkl"), 'rb'))
                nonpadded_rows = pickle.load(open(os.path.join(exp_path, run_name, "model_outputs", "nonpadded_rows.pkl"), 'rb'))
                #attributions = pickle.load(open(os.path.join(exp_path, run_name, "model_outputs", "attributions.pkl"), 'rb'))
                for subject_split in subject_splits:
                    for subject in run_splits[subject_split]:
                        for t_ix in range(len(targets[subject])):
                            if nonpadded_rows[subject][t_ix] > 0:
                                data.append({
                                    "subject": subject,
                                    "t_ix": t_ix,
                                    "target": targets[subject][t_ix],
                                    "pred": predictions[subject][t_ix],
                                    #"attribution": attributions[subject][t_ix],
                                    "exp": exp_better_names.get(exp_name, exp_name),
                                    "seed": seed,
                                    "split": subject_split
                                })
    return pd.DataFrame(data)
