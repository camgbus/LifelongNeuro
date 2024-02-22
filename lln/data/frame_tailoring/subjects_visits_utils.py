"""Perform different operations on the subjects_df and visits_df.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm
from collections import OrderedDict

def filter_subjects_by_visits(subjects_df, visits_df):
    '''Filter subjects for which there are no visits'''
    return subjects_df[subjects_df.apply(lambda x: 
        len(visits_df[visits_df['subject'] == x['subject']]) > 0, axis=1)]

def filter_subjects_by_list(subjects_df, subject_list):
    '''Filter subjects, leaving only those with entries in the list'''
    return subjects_df[subjects_df.subject.isin(subject_list)]

def filter_visits_by_subjects(subjects_df, visits_df):
    '''Filter visits for which there are no subjects'''
    subjects = list(subjects_df['subject'])
    return visits_df[visits_df.subject.isin(subjects)]

def filter_subjects_visits_by_list(subjects_df, visits_df, visits=None):
    '''Leave only subjects for which there is data for all the specified visits'''
    complete_subjects = []
    for subject in tqdm(set(visits_df['subject'])):
        subject_visits_df = visits_df.loc[(visits_df['subject'] == subject)]
        eventnames = set(subject_visits_df['visit_id'])
        if all([x in eventnames for x in visits]):
            complete_subjects.append(subject)
    complete_visits_df = visits_df[visits_df['subject'].isin(complete_subjects)]
    complete_visits_df = complete_visits_df[complete_visits_df['visit'].isin(visits)]
    complete_subjects_df = filter_subjects_by_visits(subjects_df, complete_visits_df)
    assert len(complete_visits_df) == len(visits)*len(complete_subjects_df)
    return complete_subjects_df, complete_visits_df

def filter_by_complete_values(df, columns):
    '''Filter a subjects_df or visits_df by ensuring there are no missing values in some columns'''
    return df[df[[columns]].notnull().all(1)]

def add_subject_vars(subjects_df, visits_df, columns, mode='first', remove_from_visits=True):
    '''Add additional information to the subjects dataframe. If subjects_df == None, it is being 
    created for the first time. Possible modes to determine the value if there are several non-NaN 
    values are 'first', 'last', 'mean' or 'none'. If 'none', when there are inconsistent values, the
    value np.nan is used.
    
    Parameters:
        subjects_df (pandas.DataFrame): Subjects dataframe, or None
        visits_df (pandas.DataFrame): Visits dataframe
        col (str): Column
        mode (str): 'first', 'last', 'mean' or 'nan'
    Returns:
        subjects_df (pandas.DataFrame)
        visits_df (pandas.DataFrame) where col has been (optionally) filtered out
    '''
    if subjects_df is None:
        subject_ids = list(OrderedDict.fromkeys(list(visits_df['subject'])))
        subjects_df = pd.DataFrame(data={'subject': subject_ids})
    else:
        subject_ids = subjects_df['subject']
    
    new_df = []
    for subject_id in tqdm(subject_ids):
        subject_df = visits_df[visits_df['subject'] == subject_id] 
        new_row = [subject_id]
        for col in columns:
            values = list(subject_df[col].dropna())
            if len(values) < 1:
                new_row.append(np.nan)
            elif len(set(values)) == 1:
                new_row.append(values[0])
            else:
                if mode == 'first':
                    new_row.append(values[0])
                elif mode == 'last':
                    new_row.append(values[-1])
                elif mode == 'mean':
                    new_row.append(np.mean(values))
                else:
                    new_row.append(np.nan)
        new_df.append(new_row)
    new_df = pd.DataFrame(new_df, columns=['subject'] + columns)
    new_subjects_df = pd.merge(subjects_df, new_df, on=["subject"])
    if remove_from_visits:
        visits_df = visits_df.drop(columns, axis=1)
    return new_subjects_df, visits_df

def update_variables(subjects_df, visits_df, variables_df):
    '''For a variables df, updates all columns except for var, subject_var, group and subgroup
    The new columns are: continuous, values_range, values_dist, ratio_per_subject, ratio_per_row'''
    variables  = list(variables_df['var'])
    subject_vars = list(variables_df['subject_var'])
    groups = list(variables_df['group'])
    subgroups = list(variables_df['subgroup'])
    continuous = []
    values_range = []
    values_dist = []
    ratio_per_subject = []
    ratio_per_row = []
    
    def add_values(relevant_df, col_name):
        continuous.append(is_numeric_dtype(relevant_df[col_name]) and len(relevant_df[col_name].unique()) > 10)
        possible_values = relevant_df[col_name].dropna().unique()
        if len(possible_values)>0 and continuous[-1]:
            values_range.append((min(possible_values), max(possible_values)))
            bins = pd.cut(relevant_df[col_name].dropna(), bins=5)
            value_counts_binned = bins.value_counts().to_dict()
            value_counts_binned = [f"[{bin.left}-{bin.right}]: {count}" for bin, count in value_counts_binned.items()]
            values_dist.append(value_counts_binned)
        else:
            values_range.append(sorted(list(possible_values)))
            values_dist.append(relevant_df[col_name].value_counts().to_dict())
        ratio_per_row.append((len(relevant_df) - relevant_df[col_name].isnull().sum()) / len(relevant_df)) 
        subjects = relevant_df.subject.unique()
        ratio_per_subject.append(sum(len(relevant_df[relevant_df['subject'] == subject][col_name].dropna()) > 0 
                                       for subject in subjects) / len(subjects))
        
    for var, subject_var in tqdm(zip(variables, subject_vars), total=len(variables)):
        if subject_var:
            add_values(subjects_df, var)
        else:
            add_values(visits_df, var)
        
    variables_df = pd.DataFrame({
        'var': variables,
        'group' : groups,
        'subgroup': subgroups,
        'subject_var': subject_vars,
        'continuous': continuous,
        'values_range': values_range,
        'values_dist': values_dist,
        'ratio_per_subject': ratio_per_subject,
        'ratio_per_row': ratio_per_row
    })
    return variables_df