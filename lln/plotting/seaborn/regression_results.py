"""Plotting regression results as:
- Scatterplots of predicted vs. target values,
- Boxplots of regression errors
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from lln.plotting.seaborn.rendering import save

def plot_reg_scatters(y_true, y_pred, figure_size=(5.25,3.75), title='', xlabel='Actual', ylabel='Predicted', name='', save_path=None):
    '''Scatterplot of predicted vs. target values'''
    
    g = sns.jointplot(x=y_true, y=y_pred, kind="scatter", alpha=0.5, height=figure_size[0], marginal_kws={'bins': 20, 'fill': True})

    # Add the perfect prediction line (y = x) in light gray
    perfect_line = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100)
    g.ax_joint.plot(perfect_line, perfect_line, color='lightgray', linestyle='--')

    # Add KDE density curves on the margins
    g.plot_marginals(sns.kdeplot, color="blue", fill=True)

    # Set plot titles and labels
    g.ax_joint.set_title(title, pad=20)
    g.set_axis_labels(xlabel, ylabel)
    
    if save_path is not None:
        save(plt, path=save_path, file_name=f'RegScatter_{name}.svg')
        plt.close()
    else:
        return plt

def plot_reg_boxplots(y_true, y_pred, figure_size=(5.25,3.75), name='', save_path=None, ax=None):
    '''Boxplots showing different errors'''
    
    # Calculate absolute and squared errors
    abs_errors = np.abs(y_true - y_pred)
    squared_errors = (y_true - y_pred) ** 2

    # Create a DataFrame for plotting
    error_data = pd.DataFrame({
        'Absolute Error': abs_errors,
        'Squared Error': squared_errors
    })
    
    # Create the plot if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
    
    # Create the boxplots
    sns.boxplot(data=error_data, ax=ax)
    
    if save_path is not None:
        save(plt, path=save_path, file_name=f'RegErrors_{name}.svg')
        plt.close()
    else:
        return plt
    
def plot_reg_scatters_df(df, x_col, y_col, hue_col=None, palette=None, facet_col=None, 
                         xlabel='Actual', ylabel='Predicted', hue_label=None, facet_label=None,
                         figure_size=(5.25,3.75), title='', name='', save_path=None):
    '''Scatterplot of predicted vs. target values'''
    
    if facet_col and facet_col in df.columns:
        cat_order = sorted(list(df[facet_col].unique()))
        g = sns.FacetGrid(df, col=facet_col, col_wrap=4, height=3, aspect=1, col_order=cat_order)
        g.map_dataframe(sns.scatterplot, x=x_col, y=y_col, hue=hue_col, palette=palette, alpha=0.5)

        # Add a 45-degree line and set axes limits in each subplot
        for ax in g.axes:
            min_val = min(df[x_col].min(), df[y_col].min())
            max_val = max(df[x_col].max(), df[y_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray')
            
            # Set new limits with some padding
            ax.set_xlim(min_val*0.9, max_val*1.1)
            ax.set_ylim(min_val*0.9, max_val*1.1)

        sample_counts = df[facet_col].value_counts()

        # Set titles including the category name and sample count
        for ax, cat in zip(g.axes.flat, cat_order):
            sample_count = df[df[facet_col] == cat].shape[0]
            ax.set_title(f'{cat} (N={sample_count})')
        g.set_axis_labels(xlabel, ylabel)
        
        plt.tight_layout(pad=1.0)
        g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)

    else:
        
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette=palette, ax=ax, alpha=0.5)

        # Additional optional styling
        min_val = min(df[x_col].min(), df[y_col].min())
        max_val = max(df[x_col].max(), df[y_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray')

        # Set plot titles and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    if hue_col is not None:
        hue_counts = df[hue_col].value_counts().to_dict()
        hue_counts = {str(k): v for k, v in hue_counts.items()}
        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = [f'{label} (N={hue_counts[label]})' if label in hue_counts else label for label in labels]        
        title = hue_label if hue_label is not None else hue_col
        plt.legend(handles=handles, labels=new_labels, title=title, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    if save_path is not None:
        save(plt, path=save_path, file_name=f'RegScatter_{name}.svg')
        plt.close()
    else:
        return plt
    
def plot_reg_boxplots_df(df, true_col, pred_col, hue_col=None, palette=None, hue_label=None, 
                         facet_col=None, facet_label=None,
                         mode='Error', figure_size=(13,3), title='', name='', save_path=None):
    if mode == 'Absolute Error':
        df['Error'] = np.abs(df[true_col] - df[pred_col])
    elif mode == 'Squared Error':
        df['Error'] = (df[true_col] - df[pred_col]) ** 2
    elif mode == 'Residuals':
        df['Error'] = df[true_col] - df[pred_col]
    else:
        raise ValueError(f'Unknown mode: {mode}')
    
    hue_label = hue_label if hue_label is not None else hue_col
    facet_label = facet_label if facet_label is not None else facet_col
    if facet_col is None:
        facet_col = hue_col
        facet_label = hue_label
    
    # Create the boxplots
    fig, ax = plt.subplots(figsize=figure_size)
    #ax.set_title(f'{mode} between {true_col} and {pred_col}')
    ax.set_xlabel(facet_label)
    ax.set_ylabel(mode)
    sns.boxplot(data=df, x=facet_col, y='Error', hue=hue_col, palette=palette, ax=ax, order=sorted(df[facet_col].unique()))
    
    if hue_col is not None:
        hue_counts = df[hue_col].value_counts().to_dict()
        hue_counts = {str(k): v for k, v in hue_counts.items()}
        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = [f'{label} (N={hue_counts[label]})' if label in hue_counts else label for label in labels]
        plt.legend(handles=handles, labels=new_labels, title=hue_label, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    if save_path is not None:
        save(plt, path=save_path, file_name=f'{mode.replace(" ", "")}_{name}.svg')
        plt.close()
    else:
        return plt