import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.9)

def add_refline(plot, location=None):
    if location:
        plot.refline(x=location[0], y=location[1], color = "black", lw = 2)

def add_text(plot, text_labels = []):
    for (x, y, text) in text_labels:
        plot.ax_joint.text(x, y, text)

def highlight_legend_titles(ax, legend_titles):
    '''Make the categories of the legend bold and add a space between them.'''
    legend = ax.get_legend()
    first_item = True
    for text in legend.texts:
        if text.get_text() in legend_titles:
            if first_item:
                first_item = False
            else:
                text.set_text('\n' + text.get_text())   
            text.set_fontweight('bold')

def plot_scatter(df, x_col, y_col, hue_col, title, figsize=(5, 5), palette=None, refline=None, text_labels=[]):
    if palette:
        palette = sns.color_palette(palette)
    plt.figure(figsize=figsize)
    plot = sns.scatterplot(x=x_col,
                    y=y_col, 
                    hue=hue_col, palette=palette, #sns.color_palette("mako_r", as_cmap=True),
                    data=df)
    add_refline(plot, refline)
    add_text(plot, text_labels)
    plt.title(title, weight='bold')
    plt.legend(title=hue_col, loc='upper right', bbox_to_anchor=(1.45, 1.01))
    return plt

def plot_jointplot(df, x_col, y_col, hue_col, figsize=(10, 10), palette=None, refline=None, text_labels=[]):
    if palette:
        palette = sns.color_palette(palette)
    plot = sns.jointplot(x=x_col,
                    y=y_col, 
                    hue=hue_col, palette=palette,
                    data=df,
                    height=figsize[0])
    add_refline(plot, refline)
    add_text(plot, text_labels)
    plt.legend(title=hue_col, loc='upper right', bbox_to_anchor=(1.45, 1.2))
    return plt

def plot_regression_scatter(predictions, targets, figsize=(10, 10), title=None):
    """
    Plot a scatter plot of predictions vs targets. Called by RegressorTrainer.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=targets, y=predictions)
    max_val = max(max(predictions), max(targets))
    min_val = min(min(predictions), min(targets))
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    if title:
        plt.title(title)
    plt.tight_layout()
    return plt

def plot_violin(df, x_col, y_col, hue_col, title, figsize=(5, 5), palette=None, split=False):
    #if palette:
    #    palette = sns.color_palette(palette)
    plt.figure(figsize=figsize)
    plot = sns.violinplot(x=x_col,
                    y=y_col, 
                    hue=hue_col, palette=palette, #sns.color_palette("mako_r", as_cmap=True),
                    split=split, data=df)
    for item in plot.get_xticklabels():
        item.set_rotation(45)
    plt.title(title, weight='bold')
    plt.legend(title=hue_col, loc='upper right', bbox_to_anchor=(1.13, 1.01))
    return plt