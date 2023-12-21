"""Run experiment runs until there are no more available ones.
"""

import os
import argparse
import pandas as pd
from lln.utils.io_locking import read_pandas_df, unlock_file
from lln.exp.Experiment import ExperimentRun
from lln.utils.communication.TelegramBot import TelegramBot

def update_exp_status(exps_file, exp_name, config_name, from_status, to_status):
    '''Update the status of ongoing experiments to 'DONE' or 'FAILED'.'''
    file, df = read_pandas_df(exps_file)
    df.loc[(df['exp'] == exp_name) & (df['config'] == config_name) & (df['status'] == from_status), ['status', 'end']] = [to_status, pd.Timestamp.now()]
    unlock_file(file)
    df.to_csv(exps_file, index=False)

def get_next_exp(exps_file, hardware_name):
    '''Returns the next experiment to run, or None if there are no more experiments to run.'''
    file, df = read_pandas_df(exps_file)
    ready_rows = df[df['status'] == 'READY']
    if ready_rows.empty:
        return None
    next_exp = df[df['status'] == 'READY'].iloc[0]
    exp_name, config_name, split, seed = next_exp['exp'], next_exp['config'], next_exp['split'], next_exp['seed']
    df.loc[(df['exp'] == exp_name) & (df['config'] == config_name) & (df['split'] == split) & (df['seed'] == seed), ['status', 'hardware', 'start']] = ['RUNNING', hardware_name, pd.Timestamp.now()]
    unlock_file(file)
    df.to_csv(exps_file, index=False)
    return (exp_name, config_name, split, seed)

def run(args):
    
    hardware_name = args.get('hardware_name', 'pc')
    debugging = args.get('debugging', False)

    # Initialize communicator
    if args.get('communicator', 'telegram_bot') == 'telegram_bot':
        com = TelegramBot()

    # Collect available experiment runs and update the file
    exps_path = args.get('exps_path')
    exps_file = os.path.join(exps_path, 'exps_summary.csv')    
    next_exp = get_next_exp(exps_file, hardware_name)
    failures = 0
    while next_exp is not None:
        exp_name, config_name, split, seed = next_exp
        exp_run = ExperimentRun(exps_path, exp_name, config_name, split=split, seed=seed, debugging=debugging)
        if debugging:
            exp_run.run(split, seed)
        else:
            com.send_msg(f'Starting {exp_name} run {split}, {seed} in {hardware_name}')
            try:
                exp_run.run(split, seed)
                exp_run.finish()
                update_exp_status(exps_file, exp_name, config_name, from_status='RUNNING', to_status='DONE')
            except Exception:
                exp_run.finish(failed=True)
                failures += 1
                update_exp_status(exps_file, exp_name, config_name, from_status='RUNNING', to_status='FAILED')
            com.send_msg(f'Finished {exp_name} run {split}, {seed} in {hardware_name} with {exp_run.summary[config_name]["state"]}')

        # Break to prevent chain failures
        if failures > 0:
            next_exp = None
        else:
            next_exp = get_next_exp(exps_file, hardware_name)

    if failures > 0:
        print(f'Finishing in {hardware_name} due to failures.')
        com.send_msg(f'Finishing in {hardware_name} due to failures.')
    else:
        print(f'Finishing in {hardware_name} (no more experiments).')
        com.send_msg(f'Finishing in {hardware_name} (no more experiments).') 

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
