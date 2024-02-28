"""Run experiment runs until there are no more available ones.
"""

import os
import argparse
import pandas as pd
from lln.utils.io_locking import read_pandas_df, unlock_file
from lln.exp.Experiment import ExperimentRun
from lln.utils.communication.TelegramBot import TelegramBot

def run(args):
    '''Run a single experiment run.
    
    Args:
        exps_path (str): Path to the experiments folder.
        hardware_name (str): Name of the hardware running the experiments.
        communicator (str): Name of the communicator to use, e.g. telegram_bot
        debugging (bool): Whether to run in debugging mode. In the debugging mode, no messages are 
            sent through the communicator and instead of saving the stdout and potential traceback, 
            they are printed to the console. Also, the state is kept as 'FAILED' instead of 'RUNNING'.
    '''
    
    hardware_name = args.get('hardware_name', 'pc')
    debugging = args.get('debugging', False)
    ready_state = 'READY' if not debugging else 'FAILED'
    
    # Initialize communicator
    if args.get('communicator', 'telegram_bot') == 'telegram_bot':
        com = TelegramBot()

    # Collect available experiment runs and update the file
    exps_path = args.get('exps_path')
    exps_file = os.path.join(exps_path, 'exps_overview.csv')    
    
    next_exp = get_next_run(exps_file, hardware_name, ready_state=ready_state)
    failures = 0
    while next_exp is not None:
        exp_name, config_name, split, seed = next_exp
        exp_run = ExperimentRun(exps_path, exp_name, config_name, split=split, seed=seed, debugging=debugging)
        if debugging:
            exp_run.run()
        else:
            com.send_msg(f'Starting {exp_name} run {split}, {seed} in {hardware_name}')
            try:
                exp_run.run()
                exp_run.finish()
                update_run_state(exps_file, exp_name, config_name, split, seed)
            except Exception:
                exp_run.finish(failed=True)
                failures += 1
                update_run_state(exps_file, exp_name, config_name, split, seed, to_state='FAILED')
            com.send_msg(f'Finished {exp_name} run {split}, {seed} in {hardware_name} with {exp_run.summary[config_name]["state"]}')

        # Break to prevent chain failures
        if debugging or failures > 0:
            next_exp = None
        else:
            next_exp = get_next_run(exps_file, hardware_name, ready_state=ready_state)

    if failures > 0:
        print(f'Finishing in {hardware_name} due to failures.')
        com.send_msg(f'Finishing in {hardware_name} due to failures.')
    else:
        if debugging:
            print(f'Finished debugging, finished successfully or there were no FAILED experiments.')
        else:
            print(f'Finishing in {hardware_name} (no more experiments).')
            com.send_msg(f'Finishing in {hardware_name} (no more experiments).')

def update_run_state(exps_file, exp_name, config_name, split, seed, to_state='DONE'):
    '''Update the state of ongoing experiments to 'DONE' or 'FAILED' and free up blocked runs.'''
    file, df = read_pandas_df(exps_file, column_types={'exp': str, 'config': str, 'split': str, 
        'seed': int, 'state': str, 'blocked_by': str, 'hardware': str, 'start': str, 'end': str})
    mask = (df['exp'] == exp_name) & (df['config'] == config_name) & (df['split'] == split) & (df['seed'] == seed) & (df['state'] == 'RUNNING')
    df.loc[mask, ['state', 'end']] = [to_state, pd.Timestamp.now()]
    if to_state == 'DONE':
        # Remove that dependency from the 'blocked_by' column
        mask = (df['exp'] == exp_name) & (df['split'] == split) & (df['seed'] == seed) & (df['state'] == 'BLOCKED')
        df.loc[mask, 'blocked_by'] = df.loc[mask, 'blocked_by'].apply(lambda x: '-'.join([config for config in str(x).split('-') if config != config_name]))
        df.loc[mask & (df['blocked_by'] == ''), 'state'] = 'READY'
    unlock_file(file)
    df.to_csv(exps_file, index=False)

def get_next_run(exps_file, hardware_name, ready_state='READY'):
    '''Returns the next experiment to run, or None if there are no more experiments to run.'''
    file, df = read_pandas_df(exps_file, column_types={'exp': str, 'config': str, 'split': str, 
        'seed': int, 'state': str, 'blocked_by': str, 'hardware': str, 'start': str, 'end': str})
    ready_rows = df[df['state'] == ready_state]
    to_state = 'RUNNING' if ready_state == 'READY' else 'FAILED'
    if ready_rows.empty:
        return None
    next_exp = ready_rows.iloc[0]
    exp_name, config_name, split, seed = next_exp['exp'], next_exp['config'], next_exp['split'], next_exp['seed']
    df.loc[(df['exp'] == exp_name) & (df['config'] == config_name) & (df['split'] == split) & (df['seed'] == seed), ['state', 'hardware', 'start']] = [to_state, hardware_name, pd.Timestamp.now()]
    unlock_file(file)
    df.to_csv(exps_file, index=False)
    return (exp_name, config_name, split, seed)

def parse_arguments():
    '''Parses the command line arguments.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--exps_path", action='store', type=str, required=True)
    parser.add_argument("-hn", "--hardware_name", action='store', type=str, default='pc')
    parser.add_argument("-com", "--communicator", action='store', type=str, default='telegram_bot')
    parser.add_argument("-d", "--debugging", action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    args = vars(args)
    run(args)
