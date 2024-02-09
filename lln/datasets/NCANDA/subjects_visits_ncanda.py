"""Loader for the NCANDA dataset, from the summary.csv files.
The data should be in the following structure:
| <data_path>
|| NCANDA_SNAPS_8Y_REDCAP_V03
||| summaries
|||| redcap
||||| ataxia.csv
||||| ...
|| NCANDA_SNAPS_8Y_DIFFUSION_V01
|||| diffusion
||||| ...
||| summaries
|| NCANDA_SNAPS_8Y_STRUCTURAL_V01
||| summaries
|| NCANDA_SNAPS_8Y_RESTINGSTATE_V01
||| summaries
|||| restingstate
"""
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import lln.utils.io as io
from lln.data.data_loading.subjects_visits import SubjectsVisitsLoader
from lln.datasets.NCANDA.variable_definitions import VARS, PATHS
from lln.data.frame_tailoring.subjects_visits_utils import add_subject_vars, filter_visits_by_subjects

class NCANDALoader(SubjectsVisitsLoader):
    ''''Loads the NCANDA data into data frames'''
        
    def read_data(self):
        # Load all the variables in the visits_df
        variables_cols = ["var", "group", "subgroup", "continuous", "range", "derived_from"]
        variables = []
        visits_df = None
        for group in ["tabular", "structural", "functional", "diffusion"]:
            for subgroup, var_subgroup in VARS[group].items():
                for file, var_list in var_subgroup:
                    if var_list:
                        old_names_cols = [old_name for old_name, _ in var_list]
                        rows = io.load_df(os.path.join(PATHS[group], file+'.csv'), sep =',', cols=["subject", "visit"]+old_names_cols)
                        rows = rows.rename(columns={old_name: new_name for old_name, new_name in var_list})
                        if visits_df is None:
                            visits_df = rows
                        else:
                            visits_df = visits_df.merge(rows, on=["subject", "visit"])
                        # Collect variable information
                        for _, col_name in var_list:
                            variable_info = [col_name, group, subgroup, is_numeric_dtype(visits_df[col_name])]
                            possible_values = visits_df[col_name].dropna().unique()
                            if len(possible_values)>0 and variable_info[-1]:
                                variable_info.append((min(possible_values), max(possible_values)))
                            else:
                                variable_info.append(sorted(list(possible_values)))
                            variables.append(variable_info+[None])
        # Rename variables, and replace visit names
        visits_df = visits_df.rename(columns={"subject": "subject_id", "visit": "visit_id"})
        visits_df = visits_df.replace({'visit_id': {"baseline": "year_0", "followup_1y": "year_1", "followup_2y": "year_2", "followup_3y": "year_3", "followup_4y": "year_4", "followup_5y": "year_5", "followup_6y": "year_6", "followup_7y": "year_7", "followup_8y": "year_8"}})
        # Remove the subject-wise variables from visits_df and place them on the subjects_df, 
        # checking for consistency
        print(f"There were initially {len(set(list(visits_df['subject_id'])))} subjects and {len(visits_df)} visits.")
        subjects_df, visits_df = add_subject_vars(None, visits_df, ["family_id", "race_ethnicity", "sex_at_birth", "site_id"], mode="nan")
        subjects_df = subjects_df.dropna()
        visits_df = filter_visits_by_subjects(subjects_df, visits_df)
        print(f"After removing inconsistent subject rows, there are {len(subjects_df)} subjects and {len(visits_df)} visits.")
        return subjects_df, visits_df, pd.DataFrame(variables, columns=variables_cols)
