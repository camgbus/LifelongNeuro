"""Match subjects from two groups using Nearest Neighbor Matching. For the smaller group 1, find one
after the other the nearest neighbor in group 2. If specified, remove the selected neighbor from 
group 2 so that the two groups have a similar number of samples.
"""
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

def match_subjects(df_a, df_b, avoid_double_subject_matching=True):
    if len(df_a) < len(df_b):
        df_small, df_large = df_a, df_b
    else:
        df_small, df_large = df_b, df_a

    # Initialize NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    
    # Fit the model on the smallest group
    nn.fit(df_small)

    # Find the nearest neighbors in the large group for each item in the small group
    distances, indices = nn.kneighbors(df_large)

    # Matched pairs
    matched_pairs = [(i, j[0]) for i, j in enumerate(indices)]

    # Print matched pairs
    print("Matched pairs:")
    for pair in matched_pairs:
        print("Group 1 index:", pair[0], "| Group 2 index:", pair[1])