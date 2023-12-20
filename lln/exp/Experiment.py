"""Experiment class that tracks experiments. A directory is created for the experiment. There,
a batch script and/or configuration can be stored, along with intermediate and results files.
Once the Experiment is finished for a split, metadata such as duration and error stack traces are 
stored. The output printed to the console is likewise stored.
"""

import os
import sys
import io
import traceback
from lln.utils.helper_functions import get_time, get_time_string
from lln.utils.io import load_json, dump_json

class Experiment:
    '''An Experiment that creates a directory with its name and stores a config.json file.
    Parameters:
        config (dict): a dictionary with parameters that will be passed to the starting function.
            If a 'name' is defined, it will be used as experiment name. Otherwise, a datestring is 
            used. The experiment directory has the name of the experiment.
        exps_path (str): path to the directory where the experiment directory will be created.
        notes (str): optional notes about the experiment that are stored as a notes.txt file.
    '''
    def __init__(self, config, exps_path, notes=''):
        if 'exp_name' not in config:
            config['exp_name'] = get_time_string(get_time())
        self.exp_name = config['exp_name']
        self.path = os.path.join(exps_path, self.exp_name)
        assert not os.path.exists(self.path), "Experiment {} already exists".format(self.path)
        os.makedirs(self.path)
        self.config = config
        dump_json(self.config, path=self.path, file_name='config')
        if notes:
            with open(os.path.join(self.path, "notes.txt"), 'w', encoding="utf-8") as f:
                f.write(notes)
                
    def average_runs(self, k_folds=None, seed=0, run_names=None):
        '''Summarizes the results of multiple experiment runs, e.g. splits or seeds. Each run should 
        have a subdirectory with the corresponding name. The summary is stored in the experiment 
        directory. Summarized files are created with the suffix '_'.join(run_names).'''
        if k_folds is not None:
            assert run_names is None and seed is not None
            run_names = [f'SPLIT_{split}_SEED_{seed}' for split in range(k_folds)]
        for run_name in run_names:
            run_path = os.path.join(self.path, run_name)
            assert os.path.exists(run_path), "Experiment {}, run {} not found".format(self.exp_name, run_name)
            
        # TODO actually average files
        
        return False
    
class ExperimentRun:
    '''An Experiment subdirecory with a specific split and seed.
    
        config (dict): a dictionary with parameters that will be passed to the starting function.
            If a 'name' is defined, it will be used as experiment name. Otherwise, a datestring is 
            used. The experiment directory has the name of the experiment.
        output_path (str): path to the directory where the experiment directory will be created.
    '''
    def __init__(self, exp_name, exps_path, split=0, seed=0, debugging=False, continue_run=False):
        self.start_time = get_time()
        self.debugging = debugging
        if not debugging:
            self.old_stdout = sys.stdout
            sys.stdout = self.stdout = io.StringIO()
        self.exp_name = exp_name
        self.exp_path = os.path.join(exps_path, exp_name)
        assert os.path.exists(self.exp_path), f"Experiment not found in {self.exp_path}"
        self.run_name = f'SPLIT_{split}_SEED_{seed}'
        self.run_path = os.path.join(self.exp_path, self.run_name)
        self.config = load_json(path=self.exp_path, file_name='config')
        if continue_run:
            assert os.path.exists(self.run_path), f"Experiment {self.exp_name}, run {self.run_name} not found"
        else:
            assert not os.path.exists(self.run_path), f"Experiment {self.exp_name}, run {self.run_name} already exists"
            os.makedirs(self.run_path)
        self.summary = {'start_time': get_time_string(self.start_time)}

    def run(self, split, seed):
        ''''Runs the experiment with the given split and seed. The run_function is defined in the 
        config and should take the config, run path, split and seed as arguments.'''
        run_function_name = self.config['run_function']
        module_name, method_name = run_function_name.rsplit('.', 1)
        module = __import__(module_name, fromlist=[method_name])
        run_function = getattr(module, method_name)
        run_function(self.config, self.run_path, split, seed)

    def finish(self, failed=False, notes=''):
        '''Finishes the run by storing the summary and stdout. If failed, also stores the traceback.
        '''
        elapsed_time = get_time() - self.start_time
        if failed:
            tb = traceback.format_exc()
            self.summary['state'] = 'FAILURE'
            with open(os.path.join(self.run_path, 'traceback.txt'), 'w', encoding="utf-8") as f:
                f.write(tb)
        else:
            self.summary['state'] = 'SUCCESS'
        self.summary['elapsed_time'] = '{0:.2f} min'.format(elapsed_time.total_seconds()/60)
        self.summary['notes'] = notes
        if not self.debugging:
            sys.stdout = self.old_stdout
            with open(os.path.join(self.run_path, "stdout.txt"), 'w', encoding="utf-8") as f:
                f.write(self.stdout.getvalue())
        dump_json(self.summary, path=self.run_path, file_name='summary')

