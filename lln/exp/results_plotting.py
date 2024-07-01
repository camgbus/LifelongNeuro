'''Combine multiple experiments and/or seeds to obtain more robust predictions and uncertainty
estimates. This script is meant to be used after the experiments have been run and the results are
stored in the output folder. The script will directly read the predictions and targets.
'''
import os
import json
import pickle
import numpy as np
import pandas as pd
import pygal
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from lln.eval.metrics.classification import confusion_matrix
from lln.plotting.seaborn.reliability import plot_reliability_diagram
from lln.plotting.seaborn.confusion_matrix import plot_confusion_matrix, plot_confusion_matrix_for_timepoint
from lln.plotting.pygal.rendering import save_svg, display_html
from lln.utils.io import read_local_paths
from lln.plotting.seaborn.rendering import save

palette = {'mint': '#9AE2E0', 'light_blue': '#6AB8E8', 'purple': '#A99EEC', 'pink': '#FE8AA8', 'orange': '#FFB293', 'yellow': '#F1DD65', 'green': '#ACDD7F', 'mid_gray': '#8A8A8A', 'red': '#EAA4A4'}

def merge_in_df(exps_path, exp_names, exp_better_names = dict(), subject_splits = ['Val', 'Test']):
    data = []
    for exp_name in exp_names:
        exp_path = os.path.join(exps_path, exp_name)
        config = json.load(open(os.path.join(exp_path, "config_train.json")))
        for split in range(config['nr_splits']):
            for seed in config['seeds']:
                run_name = f"SPLIT_{split}_SEED_{seed}"
                run_splits = json.load(open(os.path.join(exp_path, run_name, "split.json"), 'rb'))
                targets = pickle.load(open(os.path.join(exp_path, run_name, "model_outputs", "targets.pkl"), 'rb'))
                predictions = pickle.load(open(os.path.join(exp_path, run_name, "model_outputs", "predictions.pkl"), 'rb'))
                nonpadded_rows = pickle.load(open(os.path.join(exp_path, run_name, "model_outputs", "nonpadded_rows.pkl"), 'rb'))
                attributions = pickle.load(open(os.path.join(exp_path, run_name, "model_outputs", "attributions.pkl"), 'rb'))
                for subject_split in subject_splits:
                    for subject in run_splits[subject_split]:
                        for t_ix in range(len(targets[subject])):
                            if nonpadded_rows[subject][t_ix] > 0:
                                data.append({
                                    "subject": subject,
                                    "t_ix": t_ix,
                                    "target": targets[subject][t_ix],
                                    "pred": predictions[subject][t_ix],
                                    "attribution": attributions[subject][t_ix],
                                    "exp": exp_better_names.get(exp_name, exp_name),
                                    "seed": seed,
                                    "split": subject_split
                                })
    return pd.DataFrame(data)

def per_experiment_confusion_matrix(df, exps_path, exp_name, exp_better_name, labels, seeds=[0], weighted=False, split_name='Test'):
    '''Joins the predictions across splits to output 1 confusion matrix on the val/test data for the
    whole experiment'''
    df_seeds = df[df['seed'].isin(seeds)]
    df_exp = df_seeds[df_seeds['exp'] == exp_better_name]
    cm = confusion_matrix(df_exp["target"], df_exp["pred"], nr_labels=len(labels))
    plot_confusion_matrix(cm, labels=labels, figure_size=(12,10), cmap='crest_r', weighted=weighted, name=split_name, save_path=os.path.join(exps_path, exp_name))

def per_experiment_per_timepoint_confusion_matrix(df, exps_path, exp_name, exp_better_name, labels, seeds=[0], weighted=False, figure_size=(25, 5), split_name='Test'):
    '''Prints multiple subplots, one for each time point'''
    df_seeds = df[df['seed'].isin(seeds)]
    df_exp = df_seeds[df_seeds['exp'] == exp_better_name]
    time_ixs = sorted(list(df_exp["t_ix"].unique()))
    plt.figure()
    for time_ix in time_ixs:
        fig, axes = plt.subplots(1, len(time_ixs), figsize=figure_size)
        for i, time_ix in enumerate(time_ixs):
            df_class_t = df_exp[df_exp['t_ix'] == time_ix]
            cm = confusion_matrix(df_class_t["target"], df_class_t["pred"], nr_labels=len(labels))
            ax = plot_confusion_matrix_for_timepoint(cm, labels=labels, cmap='crest_r', weighted=weighted, name=f"Time {time_ix}", ax=axes[i])
            ax.set_title(f"Time {time_ix}")
        save(plt, path=os.path.join(exps_path, exp_name), file_name=f'CM_t_{split_name}_W{weighted}.svg')
        plt.close()
    
def calibrated_confusion(df, target_col, pred_col, labels, fig_name, weighted=False, save_path=None):
    cm = confusion_matrix(df[target_col], df[pred_col], nr_labels=len(labels))
    plot_confusion_matrix(cm, labels=labels, figure_size=(12,10), cmap='crest_r', weighted=weighted, name=fig_name, save_path=save_path)

def per_timepoint_calibrated_confusion(df, target_col, pred_col, time_col, labels, fig_name, weighted=False, save_path=None, figure_size=(25, 5)):
    time_ixs = sorted(list(df[time_col].unique()))
    plt.figure()
    for time_ix in time_ixs:
        fig, axes = plt.subplots(1, len(time_ixs), figsize=figure_size)
        for i, time_ix in enumerate(time_ixs):
            df_t = df[df[time_col] == time_ix]
            cm = confusion_matrix(df_t[target_col], df_t[pred_col], nr_labels=len(labels))
            ax = plot_confusion_matrix_for_timepoint(cm, labels=labels, cmap='crest_r', weighted=weighted, name=f"Time {time_ix}", ax=axes[i])
            ax.set_title(f"Time {time_ix}")
        save(plt, path=save_path, file_name=f'CM_t_{fig_name}_W{weighted}.svg')
        plt.close() 

def per_experiment_feature_attributions(df, exps_path, exp_name, exp_better_name, labels, seeds=[0], only_matching=None, split_name='Test'):
    '''Plots the feature attributions for each class in the val/test data for the whole experiment'''
    df_seeds = df[df['seed'].isin(seeds)]
    df_exp = df_seeds[df_seeds['exp'] == exp_better_name]
    CAT_COLORS = {'0': palette['yellow'], '1': palette['orange'], '2': palette['pink'], '3': palette['purple'], '4': palette['light_blue']}
    custom_style = pygal.style.Style(
        colors=tuple([CAT_COLORS[str(split_ix)] for split_ix in range(5)])
        )
    # Read what those variables are for the experiment
    exp_path = os.path.join(exps_path, exp_name)
    config_path = os.path.join(exp_path, "config_train.json")
    config = json.load(open(config_path))
    paths = read_local_paths(config["paths_name"])[config["dataset"]]
    df_name = config["df_name"]
    df_path = os.path.join(paths["output"], "subjects_visits")
    variables_df = pd.read_csv(os.path.join(df_path, f"variables_{df_name}.csv"), sep=',')
    feature_cols = variables_df[variables_df["subgroup"].isin(config["feature_subgroups"])]["var"].tolist()
    feature_subgroups = variables_df[variables_df["subgroup"].isin(config["feature_subgroups"])]["subgroup"].tolist()
    feature_subgroups = {feature_cols[ix]: sg for ix, sg in enumerate(feature_subgroups)}
    feature_cols = [col for col in feature_cols if col not in config["exclude_features"]]
    subgroups = config["feature_subgroups"]
    time_ixs = sorted(list(df_exp["t_ix"].unique()))
    for class_ix, label in enumerate(labels):
        df_class = df_exp[df_exp['target'] == class_ix]
        if only_matching == 0:
            df_class = df_class[df_class['target'] != df_class['pred']]
        if only_matching == 1:
            df_class = df_class[df_class['target'] == df_class['pred']]
        # Actually fetch attributions, the first dim is the time_ix, the second the subgroup
        feature_attributions_dict = defaultdict(lambda: defaultdict(list))        
        for time_ix in time_ixs:
            df_class_t = df_class[df_class['t_ix'] == time_ix]
            for attribution in df_class_t["attribution"]:
                assert len(attribution) == len(feature_cols)
                for ix, col in enumerate(feature_cols):
                    subgroup = feature_subgroups[col]
                    feature_attributions_dict[time_ix][subgroup].append(abs(attribution[ix]))
        # Replace each list with the mean
        for time_ix in time_ixs:
            for subgroup in feature_subgroups.values():
                if subgroup not in feature_attributions_dict[time_ix]:
                    # If there are subgroups that are not in the attributions, add them with 0
                    feature_attributions_dict[time_ix][subgroup] = 0
                else:
                    feature_attributions_dict[time_ix][subgroup] = np.mean(feature_attributions_dict[time_ix][subgroup])

        # Plot chart
        bar_chart = pygal.Bar(x_label_rotation=45, style=custom_style)
        title = f'Feature attributions for {split_name}-{label}'
        if only_matching == 0:
            title += ' for incorrect preds'
        elif only_matching == 1:
            title += ' for correct preds'
        bar_chart.title = title
        bar_chart.x_labels = subgroups
        for time_ix in time_ixs:
            per_class_values = [feature_attributions_dict[time_ix][subgroup] for subgroup in subgroups]
            bar_chart.add(str(time_ix), per_class_values)

        save_svg(bar_chart, exp_path, f"{title.replace(' ', '_')}.svg")

def per_experiment_calibration(df, exps_path, exp_name, exp_better_name, labels, seeds=[0]):
    '''Calibration plot combining multiple seeds'''
    df_seeds = df[df['seed'].isin(seeds)]
    df_exp = df_seeds[df_seeds['exp'] == exp_better_name]
    targets, predictions, uncertainties = [], [], []
    for subject in df_exp['subject'].unique():
        subject_df = df_exp[df_exp['subject'] == subject]
        for t_ix in subject_df['t_ix'].unique():
            t_ix_df = subject_df[subject_df['t_ix'] == t_ix]
            if len(t_ix_df) > 1:
                target = t_ix_df['target'].iloc[0]
                targets.append(target)
                preds = t_ix_df['pred'].values
                pred_mean = preds.mean()
                predictions.append(int(pred_mean))
                pred_std = preds.std()
                uncertainties.append(pred_std)
    # Normalize uncertainties between 0 and 1
    uncertainties = (uncertainties - min(uncertainties)) / (max(uncertainties) - min(uncertainties))
    plot_reliability_diagram(np.array(targets), np.array(predictions), np.array(uncertainties), name="Test", nr_bins=10, cmap='crest_r', save_path=os.path.join(exps_path, exp_name))

def plot_heatmap_divergence(df, exp_divider='seed', only_matching=None, save_path=None, split_name='Test'):
    runs = sorted(list(df[exp_divider].unique()))
    heatmap_data = {}
    max_possible = None
    for column in runs:
        heatmap_data[column] = []
        column_df = df[df[exp_divider] == column]
        for row in runs:
            row_df = df[df[exp_divider] == row]
            non_divider = 'exp' if exp_divider=='seed' else 'seed'
            merged_df = pd.merge(column_df, row_df, on=['subject', 't_ix', 'target', 'pred', non_divider], how='inner')
            if column == row:
                max_possible = len(merged_df)
            if only_matching == 0:
                merged_df = merged_df[merged_df['target'] != merged_df['pred']]
            if only_matching == 1:
                merged_df = merged_df[merged_df['target'] == merged_df['pred']]
            heatmap_data[column].append(len(merged_df))

    heatmap_df = pd.DataFrame(heatmap_data)
    sns.heatmap(heatmap_df, annot=True, cmap='crest_r', fmt='d', vmin=0, vmax=max_possible)
    title = f'Same across {split_name} {exp_divider}s'
    if only_matching == 0:
        title += ' for incorrect preds'
    elif only_matching == 1:
        title += ' for correct preds'
    plt.title(title)
    plt.xlabel(exp_divider)
    plt.ylabel(exp_divider)

    if save_path:
        output_file = os.path.join(save_path, f"{title.replace(' ', '_')}.svg")
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_successful_models(df, labels, exp_divider='exp', save_path=None, split_name='Test'):
    # TODO fix
    frequency_successful = defaultdict(lambda: defaultdict(int))
    for subject in df['subject'].unique():
        subject_df = df[df['subject'] == subject]
        for t_ix in subject_df['t_ix'].unique():
            t_ix_df = subject_df[subject_df['t_ix'] == t_ix]
            if len(t_ix_df) > 1:
                correct_df = t_ix_df[t_ix_df['target'] == t_ix_df['pred']]
                target = t_ix_df['target'].iloc[0]
                frequency_successful[len(correct_df)][target] += 1
    CAT_COLORS = {'0': palette['light_blue'], '1': palette['purple'], '2': palette['pink']}
    custom_style = pygal.style.Style(
        colors=tuple([CAT_COLORS[str(split_ix)] for split_ix in range(3)])
        )

    bar_chart = pygal.Bar(x_label_rotation=45, style=custom_style)
    bar_chart.title = f'Nr. successful {split_name} models'
    frequencies_successful = sorted(list(frequency_successful.keys()))
    bar_chart.x_labels = frequencies_successful
    for class_ix, label in enumerate(labels):
        per_class_values = [frequency_successful[f][class_ix] for f in frequencies_successful]
        bar_chart.add(label, per_class_values)

    if save_path:
        save_svg(bar_chart, save_path, f"successful_{split_name}_models.svg")
    else:
        display_html(bar_chart)
        

