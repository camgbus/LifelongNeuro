"""A trainer that makes longitudinal classifications.
"""

import os
import pandas as pd
import numpy as np
import itertools
import torch
from lln.training.Trainer import Trainer
from lln.eval.metrics.classification import balanced_accuracy, f1
import sys
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import sys
METRICS = {"B-Acc.": balanced_accuracy, "F1": f1}

class LongitudinalTrainer(Trainer):
    '''Trains a longitudinal classifier.'''
    def __init__(self, *args, seq_to_seq=False, labels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels
        self.seq_to_seq = seq_to_seq
        if self.metrics is None:
            self.metrics = ["B-Acc.", "F1"]
        
    def eval(self, model, eval_dataloaders, epoch_ix, verbose=False):
        '''Evaluates a model w.r.t. given metrics. Prints and saves this progress.'''
        model.eval()
        progress_summary = dict()
        for dataloader_name, dataloader in eval_dataloaders.items():
            progress_summary[dataloader_name] = dict()
            nr_batches = len(dataloader)
            total_loss = 0
            targets, predictions, nonpadded_rows = [], [], []
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = model(X)
                    total_loss += self.loss_f(pred, y).item()
                    targets += list(y.detach().cpu().numpy())
                    predictions += list(pred.argmax(1).detach().cpu().numpy())
                    if self.seq_to_seq:
                        # Set to False the rows that are padding, recognized but having all features == 0
                        nonpadded_rows += list(np.any(X.cpu().numpy() != 0, axis=2))
                    
            total_loss /= nr_batches
            # This trainer only stores the total loss
            self.loss_trajectory.append([epoch_ix, dataloader_name, float(total_loss)])
            if self.seq_to_seq:
                targets = np.concatenate(targets)
                predictions = np.concatenate(predictions)
                nonpadded_rows = np.concatenate(nonpadded_rows)
                targets, predictions = targets[nonpadded_rows], predictions[nonpadded_rows]
                
            self.progress.append([epoch_ix, dataloader_name] + [
                METRICS[metric_name](targets, predictions) for metric_name in self.metrics])
            self.plot_confusion_matrix(targets, predictions, file_name="CM_{}_{}".format(epoch_ix, dataloader_name))
            # Summarize values into progress for printing
            if verbose:
                for loss_name in self.losses:
                    progress_summary[dataloader_name][loss_name] = float(total_loss)
                for metric_name in self.metrics:
                    score = METRICS[metric_name](targets, predictions)
                    progress_summary[dataloader_name][metric_name] = score
        if verbose:
            self.print_progress(epoch_ix, progress_summary)