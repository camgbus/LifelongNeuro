"""Fill in missing values in a visits dataframe.
"""
from tqdm import tqdm

def fill_missing_values_per_subject(visits_df, features, method='interpolation'):
    '''
    Fill in missing values in a visits dataframe. For each subject, the missing values are filled in 
    with the other visits of the individual. If there are no other visits, the missing values are 
    filled in with the mean or median of the variable.
    
    Parameters:
        df (pandas.DataFrame): A Pandas df
        features (str): A list of features
        method (str): The method to use for filling in missing values. Options are 'interpolation', 
            'nearest', 'mean' and 'median'. If the variable is categorical, the mode is used.
    Returns:
        new_df (pandas.DataFrame): A Pandas df with the missing values filled in.
    '''
    # First avoid iterating over columns without missing values
    features = [x for x in features if visits_df[x].isnull().any()]
    is_categorical = {x: visits_df[x].dtype == 'object' for x in features}
    visits_df = visits_df.sort_values('visit_age')
    # Select the visits for each subject
    for subject in tqdm(visits_df['subject'].unique()):
        subject_visits = visits_df[visits_df['subject'] == subject].copy()
        for feature in features:
            if is_categorical[feature]:
                # If there are values available for that subject
                if not subject_visits[feature].isnull().all():
                    subject_visits[feature].fillna(subject_visits[feature].mode()[0], inplace=True)
                else:
                    subject_visits[feature].fillna(visits_df[feature].mode()[0], inplace=True)
            
            else:
                if not subject_visits[feature].isnull().all():
                    if method == 'interpolation':
                        # Fill in the first visit
                        subject_visits[feature].bfill(inplace=True) # Backwards fill
                        # Fill up following visits
                        subject_visits[feature].interpolate('linear', inplace=True)
                    elif method == 'nearest':
                        subject_visits[feature].fillna(subject_visits[feature].nearest(), inplace=True)
                    elif method == 'mean':
                        subject_visits[feature].fillna(subject_visits[feature].mean(), inplace=True)
                    else:
                        subject_visits[feature].fillna(subject_visits[feature].median(), inplace=True)
                # Must fill in with other values from the data frame
                else:
                    if method == 'mean':
                        subject_visits[feature].fillna(visits_df[feature].mean(), inplace=True)
                    else:
                        subject_visits[feature].fillna(visits_df[feature].median(), inplace=True)
            
            visits_df.loc[visits_df['subject'] == subject] = subject_visits
    non_filled_features = [x for x in features if visits_df[x].isnull().any()]
    assert len(non_filled_features) == 0, f'Features {non_filled_features} still have missing values'
    return visits_df
    
            