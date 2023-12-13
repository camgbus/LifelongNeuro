"""Division of subjects for cross-validation and OOD evaluation.

All subjects are divided into N groups by either scanner or acquisition site. Each of the N groups 
is further divided into K folds, keeping members of the same family together. Individuals that have 
changed acquisition site/scanner or for which any family member was at any point (within the
specified visits) scanned in different/multiple sites is discarded.
"""

import random
from tqdm import tqdm
import lln.utils.io as io

def save_restore_visit_splits(df_name, output_path, k=5, seed=0, ensure_one_site_per_family=False):
    '''Restore splits for subjects from a df'''
    name = f"splits_{df_name}_{k}_{seed}"
    try:
        splits = io.load_json(output_path, name)
    except IOError:
        subjects_df = io.load_df(output_path, f"subjects_{df_name}.csv")
        if ensure_one_site_per_family:
            assert_one_site_per_family(subjects_df)
        sites = set(list(subjects_df['site_id']))
        splits = inter_site_splits(subjects_df, sites, k=k, seed=seed)
        io.dump_json(splits, output_path, name)
    return splits

def inter_site_splits(subjects_df, sites, k=5, seed=0):
    '''Divides subjects first by site, then randomly into k folds (keeping family members together).

    Parameters:
        subjects_df (pandas.DataFrame): Subjects dataframe
    Returns:
        site_splits ({str -> {str -> [str]}}): A dictionary linking each site ID to a k-item long 
            dict linking a split ID to a subject ID list
    '''
    random.seed(seed)
    site_splits = dict()
    for site_id in tqdm(sites):
        family_groups = []
        site_df = subjects_df.loc[subjects_df["site_id"] == site_id]
        family_ids = set(site_df["family_id"])
        for family_id in family_ids:
            family_subjects = list(site_df.loc[site_df["family_id"] == family_id]["subject_id"])
            family_groups.append(family_subjects)
        splits = {str(split_ix) : [] for split_ix in range(k)}
        # Shuffle items
        random.shuffle(family_groups)
        # After shuffling, order with the largest groups first
        family_groups.sort(key=lambda x: len(x), reverse=True)
        # Assign each item, one after another, to the group with less items
        for item in family_groups:
            smallest_group_ix = min(splits.keys(), key=lambda x: len(splits[x]))
            splits[smallest_group_ix] += item
        site_splits[site_id] = splits
    return site_splits

def assert_one_site_per_family(subjects_df):
    '''Currently, there are no cases of different family members being scanned at different sites.
    Make sure this keeps being the case.
    
    Parameters:
        subjects_df (pandas.DataFrame): Subjects dataframe
    '''
    family_ids = set(subjects_df["family_id"])
    for family_id in tqdm(family_ids):
        sites = subjects_df.loc[subjects_df["family_id"] == family_id]["site_id"]
        assert len(set(sites)) == 1