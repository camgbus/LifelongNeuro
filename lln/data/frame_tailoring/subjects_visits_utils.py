"""Perform different operations on the subjects_df and visits_df.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

def filter_subjects_by_visits(subjects_df, visits_df):
    '''Filter subjects for which there are no visits'''
    return subjects_df[subjects_df.apply(lambda x: 
        len(visits_df[visits_df['subject_id'] == x['subject_id']]) > 0, axis=1)]

def filter_subjects_by_list(subjects_df, subject_ids_list):
    '''Filter subjects, leaving only those with entries in the list'''
    return subjects_df[subjects_df.subject_id.isin(subject_ids_list)]

def filter_visits_by_subjects(subjects_df, visits_df):
    '''Filter visits for which there are no subjects'''
    subject_ids = list(subjects_df['subject_id'])
    return visits_df[visits_df.subject_id.isin(subject_ids)]

def filter_subjects_visits_by_list(subjects_df, visits_df, visits=None):
    '''Leave only subjects for which there is data for all the specified visits'''
    complete_subjects = []
    for subject_id in tqdm(set(visits_df['subject_id'])):
        subject_visits_df = visits_df.loc[(visits_df['subject_id'] == subject_id)]
        eventnames = set(subject_visits_df['visit_id'])
        if all([x in eventnames for x in visits]):
            complete_subjects.append(subject_id)
    complete_visits_df = visits_df[visits_df['subject_id'].isin(complete_subjects)]
    complete_visits_df = complete_visits_df[complete_visits_df['visit_id'].isin(visits)]
    complete_subjects_df = filter_subjects_by_visits(subjects_df, complete_visits_df)
    assert len(complete_visits_df) == len(visits)*len(complete_subjects_df)
    return complete_subjects_df, complete_visits_df

def filter_by_complete_values(df, columns):
    '''Filter a subjects_df or visits_df by ensuring there are no missing values in some columns'''
    return df[df[[columns]].notnull().all(1)]

def add_subject_vars(subjects_df, visits_df, columns, mode='first', remove_from_visits=False):
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
        subject_ids = list(OrderedDict.fromkeys(list(visits_df['subject_id'])))
        subjects_df = pd.DataFrame(data={'subject_id': subject_ids})
    else:
        subject_ids = subjects_df['subject_id']
    
    new_df = []
    for subject_id in tqdm(subject_ids):
        subject_df = visits_df[visits_df['subject_id'] == subject_id] 
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
    new_df = pd.DataFrame(new_df, columns=['subject_id'] + columns)
    new_subjects_df = pd.merge(subjects_df, new_df, on=["subject_id"])
    if remove_from_visits:
        visits_df = visits_df.drop(columns, axis=1)
    return new_subjects_df, visits_df

"""
def reset_baseline_visit_from_age(visits_df, baseline_age_years, baseline_age_up_to):
    ''''Reset the baseline visit based on the age of participants, e.g. the smallest after
    baseline_age_years up to baseline_age_up_to is now the 'baseline' and the followup_<X>y 
    visit_ids are incremented'''
    new_visits_df = pd.DataFrame(data={}, columns = visits_df.columns)
    for subject_id in set(visits_df['subject_id']):
        subject_visits_df = visits_df.loc[(visits_df['subject_id'] == subject_id)]
        # Find the lowest age that is at or above 'subject_visits_df'
        subject_ages = sorted(list(subject_visits_df['age_years']))
        subject_ages = [x for x in subject_ages if x >= baseline_age_years]
        if subject_ages and subject_ages[0] <= baseline_age_up_to:
            # Add subject to new df, otherwise exclude
            subject_visits_df = subject_visits_df.loc[(subject_visits_df['age_years'] >= subject_ages[0])]
            kwargs = {"years_from_baseline" : lambda x: x['age_years'] - subject_ages[0]}
            subject_visits_df = subject_visits_df.assign(**kwargs)
            new_visits_df = pd.concat([subject_visits_df, new_visits_df], ignore_index=True)
    new_visits_df['visit_id'] = new_visits_df['years_from_baseline'].apply(lambda x: "followup_"+str(math.ceil(x))+"y")
    new_visits_df['visit_id'] = new_visits_df['visit_id'].replace('followup_0y', 'baseline')
    return new_visits_df
"""