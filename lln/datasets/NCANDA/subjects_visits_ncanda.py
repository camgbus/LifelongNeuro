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
import sys
import pandas as pd
from pandas.api.types import is_numeric_dtype
import lln.utils.io as io
from lln.data.data_loading.subjects_visits import SubjectsVisitsLoader
from lln.data.frame_tailoring.subjects_visits_utils import add_subject_vars, filter_visits_by_subjects, filter_subjects_by_list, update_variables

GROUP_PATHS = {'redcap': os.path.join('NCANDA_SNAPS_8Y_REDCAP_V03', 'summaries', 'redcap'),
               'structural': os.path.join('NCANDA_SNAPS_8Y_STRUCTURAL_V01', 'summaries', 'structural', 'longitudinal', 'freesurfer'),
               'functional': os.path.join('NCANDA_SNAPS_8Y_RESTINGSTATE_V01', 'summaries', 'restingstate'),
               'diffusion': os.path.join('NCANDA_SNAPS_8Y_DIFFUSION_V01', 'summaries', 'diffusion')}

class NCANDALoader(SubjectsVisitsLoader):
    ''''Loads the NCANDA data into data frames'''
        
    def read_data(self, remove_smri_abnormalities=True, lateral_features=False):
        # Load all the variables in the visits_df
        visits_df = None
        var_lst, var_groups, var_subgroups = [], dict(), dict()
        for group_key in self.variables.keys():
            if group_key == "structural" and not remove_smri_abnormalities:
                raise Warning("You are using structural data without removing the individuals with structural abnormalities at baseline")
            for subgroup_key, subgroup in self.variables[group_key].items():
                for file_name, var_list in subgroup.items():
                    file_path = os.path.join(self.paths["data"], GROUP_PATHS[group_key], file_name)
                    if var_list == "All":
                        usecols = None
                    else:
                        usecols = ["subject", "visit"]+var_list
                    rows = pd.read_csv(file_path, sep=',', usecols=usecols)
                    # Rename the columns to avoid conflicts when loading lateral structural features
                    if file_name == "lh.aparc.csv":
                        rows = rows.add_prefix('left_')
                        rows.rename(columns={"left_subject": "subject", "left_visit": "visit", "left_arm": "arm"}, inplace=True)
                    if file_name == "rh.aparc.csv":
                        rows = rows.add_prefix('right_')
                        rows.rename(columns={"right_subject": "subject", "right_visit": "visit", "right_arm": "arm"}, inplace=True)
                    # Rename the columns to avoid conflicts when loading DTI features
                    if file_name == "mori_ncanda_baseline_meet_criteria_fa_corrected_global_skeleton.csv":
                        rows = rows.add_suffix('_fa')
                        rows.rename(columns={"subject_fa": "subject", "visit_fa": "visit", "arm_fa": "arm"}, inplace=True)
                    if file_name == "mori_ncanda_baseline_meet_criteria_md_corrected_global_skeleton.csv":
                        rows = rows.add_suffix('_md')
                        rows.rename(columns={"subject_md": "subject", "visit_md": "visit", "arm_md": "arm"}, inplace=True)
                    # Drop a column from rows
                    usecols = rows.columns
                    if "arm" in usecols:
                        rows = rows.drop("arm", axis=1)
                    if visits_df is None:
                        visits_df = rows
                    else:
                        visits_df = visits_df.merge(rows, on=["subject", "visit"], how="outer")
                    # Collect variable information
                    for var in usecols:
                        if var not in ["subject", "visit", "arm"]:
                            var_lst.append(var)
                            var_groups[var], var_subgroups[var] = group_key, subgroup_key
            # Remove laterality of structural features
            if group_key == "structural" and not lateral_features:
                left_struct_vars = [col for col in visits_df.columns if col.startswith('left_')]
                right_struct_vars = [col for col in visits_df.columns if col.startswith('right_')]
                assert len(left_struct_vars) == len(right_struct_vars)
                for left_var in left_struct_vars:
                    nonlat_var = left_var.replace('left_', '')
                    visits_df[nonlat_var] = (visits_df[[f"left_{nonlat_var}", f"right_{nonlat_var}"]].sum(axis=1))/2
                    visits_df = visits_df.drop(f"left_{nonlat_var}", axis=1)
                    visits_df = visits_df.drop(f"right_{nonlat_var}", axis=1)
                    var_groups[nonlat_var] = var_groups[left_var]
                    var_subgroups[nonlat_var] = var_subgroups[left_var]
                var_lst = [x for x in var_lst if x not in left_struct_vars and x not in right_struct_vars]
                var_lst = var_lst + [x.replace('left_', '') for x in left_struct_vars]
        # Rename visit names
        visits_df = visits_df.replace({'visit': {"baseline": "year_0", "followup_1y": "year_1", "followup_2y": "year_2", "followup_3y": "year_3", "followup_4y": "year_4", "followup_5y": "year_5", "followup_6y": "year_6", "followup_7y": "year_7", "followup_8y": "year_8"}})
        # Remove the subject-wise variables from visits_df and place them on the subjects_df, 
        # checking for consistency
        print(f"There were initially {len(set(list(visits_df['subject'])))} subjects and {len(visits_df)} visits.")
        subjects_df, visits_df = add_subject_vars(None, visits_df, ["family_id", "race_label", "hispanic", "sex", "site", "ses_parent_yoe"], remove_from_visits=True, mode="nan")
        subjects_df = subjects_df.dropna()
        visits_df = filter_visits_by_subjects(subjects_df, visits_df)
        print(f"After removing inconsistent subject rows, there are {len(subjects_df)} subjects and {len(visits_df)} visits.")
        # Leave only 3 race/ethnicity categories: White/Caucasian, Hispanic, African-American/Black, Asian and Other
        subjects_df = subjects_df.replace({'race_label': {'African-American_Caucasian': 'Other',  'Asian_Pacific_Islander': 'Other',  'Asian_White': 'Other', 'Native American/American Indian': 'Other', 'NativeAmerican_Caucasian': 'Other', 'Pacific Islander': 'Other', 'Pacific_Islander_Caucasian': 'Other'}})
        subjects_df['race_ethnicity'] = subjects_df.apply(lambda row: 'Hispanic' if row['hispanic'] == 'Y' else row['race_label'], axis=1)
        subjects_df = subjects_df.drop("race_label", axis=1)
        subjects_df = subjects_df.drop("hispanic", axis=1)
        var_lst = [x for x in var_lst if x not in ["hispanic", "race_label"]]
        var_lst.insert(3, "race_ethnicity")
        var_groups["race_ethnicity"], var_subgroups["race_ethnicity"] = "redcap", "demographics"
        subject_vars = ["family_id", "race_ethnicity", "sex", "site", "ses_parent_yoe"]
        
        # Remove the individuals that have structural abnormalities at baseline
        # Leave the 808 subjects used in the baseline analysis of:
        # Pfefferbaum et al. Adolescent Development of Cortical and White Matter Structure in the NCANDA 
        # Sample: Role of Sex, Ethnicity, Puberty, and Alcohol Drinking, Cerebral Cortex, 26(10), pp 4101-21, 2016.
        if remove_smri_abnormalities:
            f = open(os.path.join(self.paths["additional"], "no_baseline_mri_anomalies.txt"), "r")
            paper_subset = f.read().split("\n")
            subjects_df = filter_subjects_by_list(subjects_df, paper_subset)
            visits_df = filter_visits_by_subjects(subjects_df, visits_df)
            print(f"After removing subjects with structural abnormalities, there are {len(subjects_df)} subjects and {len(visits_df)} visits in the dataset")
        
        # If we are using structural features, average the left and right sides
        # Update the variables dataframe
        variables_df = update_variables(subjects_df, visits_df, pd.DataFrame({"var": var_lst, "subject_var": [v in subject_vars for v in var_lst], "group": [var_groups[v] for v in var_lst], "subgroup": [var_subgroups[v] for v in var_lst]}))
        return subjects_df, visits_df, variables_df

