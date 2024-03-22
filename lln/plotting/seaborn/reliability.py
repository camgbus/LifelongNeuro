"""A reliability diagram for classification tasks.
"""

import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from lln.eval.metrics.classification import balanced_accuracy, f1

def plot_reliability_diagram(targets, predictions, uncertainties, name='',  nr_bins=10, 
                             cmap=None, save_path=None):

    sns.set_style('whitegrid')

    bin_limits = np.linspace(0, 1, nr_bins+1)
    confidences = 1 - uncertainties

    bins = np.digitize(confidences, bin_limits) - 1  # Binning based on uncertainties
    
    # Initialize arrays to store bin accuracy
    bin_accuracy = np.zeros(nr_bins)
    bin_confidence = np.zeros(nr_bins)
    bin_size = np.zeros(nr_bins)
    ece = 0.0
    
    # Calculate accuracy and confidence for each bin
    for i in range(nr_bins):
        bin_indices = np.where(bins == i)[0]
        if len(bin_indices) == 0:
            continue
        # This ccan output a warning of "y_pred contains classes not in y_true" for some bins
        bin_accuracy[i] = balanced_accuracy(targets[bin_indices], predictions[bin_indices])
        bin_confidence[i] = confidences[bin_indices].mean()  # Average predicted probability
        bin_size[i] = len(bin_indices)
        ece += np.abs(bin_confidence[i] - bin_accuracy[i]) * (bin_size[i] / len(targets))
    
    # Create a pandas DataFrame
    data = {'bin': range(nr_bins), 'accuracy': bin_accuracy, 'confidence': bin_confidence, 'Support': bin_size}
    df = pd.DataFrame(data)

    # Create a barplot
    plt.figure(figsize=(10, 6))  # Set the figsize to be wider horizontally
    
    ax = sns.barplot(data=df, x='confidence', y='accuracy', hue='Support', palette=cmap)
    
    # Place the legend outside the plot
    # Get the current legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Support", loc='upper right', bbox_to_anchor=(1.14, 1))

    # Set labels and title
    plt.xlabel('Confidence')
    plt.ylabel('Balanced accuracy')
    plt.title(f'Model Calibration, ECE: {ece:.2f}')
    # Set x-axis ticks
    plt.gca().set_xticks(range(nr_bins))
    bin_labels = np.round(bin_limits, 2)
    bin_labels = [f"{bin_labels[ix]:.1f}-{bin_labels[ix+1]:.1f}" for ix in range(len(bin_labels)-1)]    
    plt.gca().xaxis.set_ticklabels(bin_labels)
    # Set y-axis limits
    plt.ylim(0, 1.1)
    
    # Draw a line from the lower-left corner to the upper-right corner
    start_point = (0, 0)
    end_point = (len(df['confidence']) - 1, 1) # df['accuracy'].max())
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k:', color='gray')
    # Add N labels at the top of each bar
    '''
    for p in ax.patches:
        # Get the information needed to display the label
        x = p.get_x() + p.get_width() / 2  # x-coordinate for the label (center of the bar)
        y = p.get_height()  # y-coordinate for the label (top of the bar)
        value = p.get_height()  # The height of the bar is the value we'll display
        
        # Use ax.text to place the value at the top of the bar
        ax.text(x, y, f'{value}', ha='center', va='bottom')  # Adjust ha and va to align
    '''
    
    if save_path:
        output_file = os.path.join(save_path, f"calibration_{name}.svg")
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return ece