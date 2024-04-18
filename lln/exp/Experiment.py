"""Experiment class that tracks experiments. A directory is created for the experiment if needed. 
There, a batch script and/or configuration can be stored, along with intermediate and results files.
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
    '''An Experiment stores a config.json file in a directory with its same name.
    Parameters:
        exps_path (str): path to the directory where the experiment directory will be created.
        exp_name (str): name of the experiment, which is the name of the directory.
        config (dict): a dictionary with parameters that will be passed to the starting function.
    '''
    def __init__(self, exps_path, exp_name, config):
        # Create the experiment directory, if it does not exist
        self.exp_name = exp_name
        self.path = os.path.join(exps_path, self.exp_name)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        # Store the config
        self.config = config
        dump_json(self.config, path=self.path, file_name=f'config_{config["config_name"]}')

class ExperimentRun:
    '''An Experiment subdirecory with a specific config, split and seed.
    
        config (dict): a dictionary with parameters that will be passed to the starting function.
            If a 'name' is defined, it will be used as experiment name. Otherwise, a datestring is 
            used. The experiment directory has the name of the experiment.
        output_path (str): path to the directory where the experiment directory will be created.
    '''
    def __init__(self, exps_path, exp_name, config_name, split=0, seed=0, debugging=False, run_name=None):
        self.start_time = get_time()
        self.debugging = debugging
        if not debugging:
            self.old_stdout = sys.stdout
            sys.stdout = self.stdout = io.StringIO()
        self.exp_name = exp_name
        self.exp_path = os.path.join(exps_path, exp_name)
        assert os.path.exists(self.exp_path), f"Experiment not found in {self.exp_path}"
        self.split = split
        self.seed = seed
        if run_name is None:
            self.run_name = f'SPLIT_{split}_SEED_{seed}'
        else:
            self.run_name = run_name
        self.run_path = os.path.join(self.exp_path, self.run_name)
        if not os.path.exists(self.run_path):
            os.makedirs(self.run_path)
        self.config_name = config_name
        self.config = load_json(path=self.exp_path, file_name=f'config_{config_name}')
        # Update summary for this run
        summary_path = os.path.join(self.run_path, 'summary.json')
        if os.path.exists(summary_path):
            self.summary = load_json(path=self.run_path, file_name='summary')
        else:
            self.summary = {}
        self.summary[config_name] = {'start_time': get_time_string(self.start_time)}
        
    def run(self):
        ''''Runs the experiment with the given split and seed. The run_function is defined in the 
        config and should take the config, run path, split and seed as arguments.'''
        run_function_name = self.config['run_function']
        module_name, method_name = run_function_name.rsplit('.', 1)
        module = __import__(module_name, fromlist=[method_name])
        run_function = getattr(module, method_name)
        run_function(self.config, self.run_path, self.split, self.seed)

    def finish(self, failed=False):
        '''Finishes the run by storing the summary and stdout. If failed, also stores the traceback.
        '''
        elapsed_time = get_time() - self.start_time
        if failed:
            tb = traceback.format_exc()
            self.summary[self.config_name]['state'] = 'FAILED'
            with open(os.path.join(self.run_path, f'traceback_{self.config_name}.txt'), 'w', encoding="utf-8") as f:
                f.write(tb)
        else:
            self.summary[self.config_name]['state'] = 'DONE'
        self.summary[self.config_name]['elapsed_time'] = '{0:.2f} min'.format(elapsed_time.total_seconds()/60)
        if not self.debugging:
            sys.stdout = self.old_stdout
            with open(os.path.join(self.run_path, f'stdout_{self.config_name}.txt'), 'w', encoding="utf-8") as f:
                f.write(self.stdout.getvalue())
        dump_json(self.summary, path=self.run_path, file_name='summary')
