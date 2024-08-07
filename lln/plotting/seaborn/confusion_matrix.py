"""A confusion matrix for classification tasks.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lln.plotting.seaborn.rendering import save

def plot_confusion_matrix(cm, labels=None, figure_size=(5.25,3.75), vmin=None, vmax=None, cmap=None, 
        title='', xlabel='Predicted', ylabel='Actual', weighted=False, name='', save_path=None, ax=None):
    '''Receives a confusion matrix where the in the first dimension are the true classes and in the
    second, what they were predicted as. For instance cm[0] are the values for the true class 0
    
    If weighted=True, the values are normalized by the total actual values in each class
    '''
    if weighted:
        row_sums = np.sum(cm, axis=1)
        cm = cm / row_sums[:, np.newaxis]
        cm = cm * 100
    df = pd.DataFrame(cm, columns=labels)
    if labels is None:
        labels = [str(ix) for ix in range(len(cm))]
    df.index = labels
    plt.figure()
    sns.set_theme(rc={'figure.figsize':figure_size})
    
    if ax is not None:
        ax = sns.heatmap(df, annot=not weighted, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, fmt='g')
    else:
        ax = sns.heatmap(df, annot=not weighted, vmin=vmin, vmax=vmax, cmap=cmap, fmt='g')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title)
    
    if weighted:
        # Add percentage sign to heatmap values
        for _, value in np.ndenumerate(cm):
            ax.text(_[1] + 0.5, _[0] + 0.5, f'{value:.1f}%', ha='center', va='center', color='white')
    
    if save_path is not None:
        save(plt, path=save_path, file_name=f'CM_{name}_W{weighted}.svg')
        plt.close()
    else:
        return plt
    
def plot_confusion_matrix_for_timepoint(cm, labels=None, vmin=None, vmax=None, cmap=None, 
        title='', xlabel='Predicted', ylabel='Actual', weighted=False, name='', ax=None):
    if weighted:
        row_sums = np.sum(cm, axis=1)
        row_sums = np.where(row_sums == 0, 0.001, row_sums)
        cm = cm / row_sums[:, np.newaxis]
        cm = np.where(row_sums[:, np.newaxis] == 0, 0, cm)
        cm = cm * 100
    df = pd.DataFrame(cm, columns=labels)
    if labels is None:
        labels = [str(ix) for ix in range(len(cm))]
    df.index = labels
    
    ax = sns.heatmap(df, annot=not weighted, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, fmt='g')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title)
    
    if weighted:
        # Add percentage sign to heatmap values
        for _, value in np.ndenumerate(cm):
            ax.text(_[1] + 0.5, _[0] + 0.5, f'{value:.1f}%', ha='center', va='center', color='white')
    return ax