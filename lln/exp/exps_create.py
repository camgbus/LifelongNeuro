"""Create experiments with certain configurations. The config should be a dictionary with:
- exp_name: name of the experiment
- run_function: path to the function that runs the experiment logic, which takes the config, split 
    and seed as arguments
- all other arguments relevant for the run_function
"""

import os
from lln.exp.Experiment import Experiment

def create_exp(exps_path, exp_name, config):
    '''Create a new experiment from a name and config.
    
    Args:
        exps_path (str): path to the directory where the experiment directory will be created.
        exp_name (str): name of the experiment, which is the name of the directory.
        config_name (str): name of the config file to be stored in the experiment directory.
        config (dict): a dictionary with parameters that will be passed to the starting function.
    '''
    # If the exps_path does not exist, create it
    if not os.path.exists(exps_path):
        os.makedirs(exps_path)
    assert 'run_function' in config, "No run_function specified in config"
    assert 'config_name' in config, "No name specified in config"
    Experiment(exps_path, exp_name, config)


def create_exps(exps_path, exp_configs):
    '''Create new experiments from configurations. Because these are repeated, it is typically for
    succeding steps.
    
    Args:
        exps_path (str): path to the directory where the experiment directory will be created.
        exps (dict): a dictionary with experiment names as keys and a list of config names as values.
        configs (dict): a dictionary with config names as keys and a dictionary with config 
            parameters as values.
    '''
    # If the exps_path does not exist, create it
    if not os.path.exists(exps_path):
        os.makedirs(exps_path)
    for exp_name, config in exp_configs.items():
        assert 'run_function' in config, "No run_function specified in config"
        assert 'config_name' in config, "No name specified in config"
        Experiment(exps_path, exp_name, config)
