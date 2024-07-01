"""Metrics that take into account uncertainty quantification.
"""

from sklearn import metrics

def auroc(y_true, y_probs, avg_mode='macro'):
    '''Area under the ROC curve. plotting TPR (y) against FPR (x)'''
    return metrics.roc_auc_score(y_true, y_probs, average=avg_mode)

def brier(y_true, y_probs):
    '''Brier score = mean squared difference between the predicted probability and the actual outcome'''
    return metrics.brier_score_loss(y_true, y_probs)