"""Normalize a variable in some range.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize(df, features, feature_range=(-1,1), standard=False, new_feature_names=None):
    '''Normalize the features into some range.'''
    # Ensure no missing values
    assert sum(df[features].isna().sum()) == 0
    
    if new_feature_names is None:
        new_feature_names = features
    
    if standard:
        scaler = StandardScaler()  
        df[new_feature_names] = scaler.fit_transform(df[features])     
    else:
        scaler = MinMaxScaler(feature_range=feature_range)
        df[new_feature_names] = scaler.fit_transform(df[features])
    return df

def cap_var(df, y, y_new_name, percentiles = (0, 95)):
    values = df[y]
    min_x, max_x = np.percentile(values, percentiles[0]), np.percentile(values, percentiles[1])
    print((min_x, max_x))
    df[y_new_name] = df.apply(lambda row: max(min(row[y], max_x), min_x), axis=1)
    return df