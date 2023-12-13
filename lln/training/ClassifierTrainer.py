"""A trainer that records classification metrics and saves confusion matrix plots.
"""

import os
import torch
from lln.training.Trainer import Trainer
from lln.eval.metrics.classification import confusion_matrix, balanced_accuracy, f1
from lln.plotting.seaborn.confusion_matrix import plot_confusion_matrix
from lln.plotting.seaborn.rendering import save
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


METRICS = {"B-Acc.": balanced_accuracy, "F1": f1}

class ClassifierTrainer(Trainer):
    '''Trains a classifier.'''
    def __init__(self,  *args, labels=None, **kwargs):
        self.labels = labels
        super(ClassifierTrainer, self).__init__(*args, **kwargs)
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
            targets = []
            predictions = []
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = model(X)
                    total_loss += self.loss_f(pred, y).item()
                    targets += list(y.detach().cpu().numpy())
                    predictions += list(pred.argmax(1).detach().cpu().numpy())
            total_loss /= nr_batches
            # This trainer only stores the total loss
            self.loss_trajectory.append([epoch_ix, dataloader_name, float(total_loss)])
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
            
    def plot_confusion_matrix(self, targets, predictions, file_name):
        ''''Plot the confusion matrix for the given epoch.'''
        cm = confusion_matrix(targets, predictions, nr_labels=len(self.labels))
        plot = plot_confusion_matrix(cm, labels=self.labels, figure_size=(12,10))
        path = os.path.join(self.trainer_path, 'confusion_matrices')
        if not os.path.exists(path):
            os.makedirs(path)
        save(plot, path, file_name=file_name)
        plot.close()