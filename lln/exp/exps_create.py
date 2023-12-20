"""Create experiments with certain configurations. The config should be a dictionary with:
- exp_name: name of the experiment
- run_function: path to the function that runs the experiment logic, which takes the config, split 
    and seed as arguments
- all other arguments relevant for the run_function
"""

import os
from lln.exp.Experiment import Experiment

def create_exps(configs, exps_path):
    '''Create experiment directories with the given configurations.'''
    os.makedirs(exps_path, exist_ok=True)
    for config in configs:
        assert 'run_function' in config, "No run_function specified in config"
        Experiment(config, exps_path)
    