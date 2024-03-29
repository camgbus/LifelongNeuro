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
import json
import pandas as pd
from pandas.api.types import is_numeric_dtype
import lln.utils.io as io
from lln.data.data_loading.subjects_visits import SubjectsVisitsLoader
from lln.data.frame_tailoring.subjects_visits_utils import add_subject_vars, filter_visits_by_subjects, filter_subjects_by_list, update_variables

GROUP_PATHS = {'redcap': os.path.join('NCANDA_SNAPS_8Y_REDCAP_V03', 'summaries', 'redcap'),
               'redcap_additional': os.path.join('NCANDA_SNAPS_8Y_REDCAP_V03', 'summaries', 'additional'),
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
                    # Remove some "do not know" values from the variables
                    if file_name == "youthreport2_aay.csv":
                        for var in ["youthreport2_aay_set2_aay6", "youthreport2_aay_set2_aay7", "youthreport2_aay_set2_aay8", "youthreport2_aay_set3_aay12", "youthreport2_aay_set3_aay13", "youthreport2_aay_set3_aay14"]:
                            rows = rows.replace({var: {5: None}})
                    if file_name == "youthreport1_fhi.csv":
                        for var in ["youthreport1_yfhi3a_yfhi3a", "youthreport1_yfhi3a_yfhi3f", "youthreport1_yfhi4a_yfhi4a", "youthreport1_yfhi4a_yfhi4f"]:
                            rows = rows.replace({var: {2: None}})
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
            # Join the ysr and asr mental health features
            if group_key == "redcap_additional":
                new_subscales = {"anxdep": ["ysr_anxdep_raw", "asr_anxdep_raw"], "withdep": ["ysr_withdep_raw", "asr_withdrawn_raw"], "somatic": ["ysr_somatic_raw", "asr_somatic_raw"], "thought": ["ysr_thought_raw", "asr_thought_raw"], 
                                 "attention": ["ysr_attention_raw", "asr_attention_raw"], "aggressive": ["ysr_aggress_raw", "asr_aggressive_raw"], "rulebreak": ["ysr_rulebrk_raw", "asr_rulebreak_raw"]}
                for new_subscale, old_subscales in new_subscales.items():
                    # Note that only one value is present at once
                    visits_df[new_subscale] = visits_df[old_subscales].mean(axis=1)
                    var_groups[new_subscale], var_subgroups[new_subscale] = var_groups[old_subscales[0]], var_subgroups[old_subscales[0]]
                    var_lst.append(new_subscale)
                    var_lst = [x for x in var_lst if x not in old_subscales]
        print(f"There were initially {len(set(list(visits_df['subject'])))} subjects and {len(visits_df)} visits.")
        # Rename visit names
        visits_df = visits_df.replace({'visit': {"baseline": "year_0", "followup_1y": "year_1", "followup_2y": "year_2", "followup_3y": "year_3", "followup_4y": "year_4", "followup_5y": "year_5", "followup_6y": "year_6", "followup_7y": "year_7", "followup_8y": "year_8"}})
        # Rename all other columns
        variable_names = json.load(open(os.path.join(self.paths["additional"], "varsets", "variable_names.json"), "r"))
        visits_df = visits_df.rename(columns=variable_names)
        new_var_list = []
        for var in var_lst:
            if var in variable_names.keys():
                var_groups[variable_names[var]], var_subgroups[variable_names[var]] = var_groups[var], var_subgroups[var]
                new_var_list.append(variable_names[var])
            else:
                new_var_list.append(var)
        var_lst = new_var_list
        # Remove the individuals that have structural abnormalities at baseline
        # Leave the 808 subjects used in the baseline analysis of:
        # Pfefferbaum et al. Adolescent Development of Cortical and White Matter Structure in the NCANDA 
        # Sample: Role of Sex, Ethnicity, Puberty, and Alcohol Drinking, Cerebral Cortex, 26(10), pp 4101-21, 2016.
        if remove_smri_abnormalities:
            f = open(os.path.join(self.paths["additional"], "no_baseline_mri_anomalies.txt"), "r")
            subject_subset = f.read().split("\n")
            visits_df = visits_df[visits_df['subject'].isin(subject_subset)]
            print(f"After removing subjects with structural abnormalities, there are {len(visits_df['subject'].unique())} subjects and {len(visits_df)} visits in the dataset")
        else:
            if "structural" in self.variables.keys():
                print("Warning: You are using structural data without removing the individuals with structural abnormalities at baseline")        
        # Remove the subject-wise variables from visits_df and place them on the subjects_df
        # checking for missing values or inconsistencies
        subjects_df, visits_df = add_subject_vars(None, visits_df, ["family_id", "sex", "site"], remove_from_visits=True, mode="nan")
        missing_inconsistent_subjects = subjects_df[subjects_df.isnull().any(axis=1)]
        # There are no inconsistent subjects regarding family_id, race_label, hispanic or xex. But 
        # there are 25 subjects that have changed sites, which we will remove to perform OOD validation
        print(f"There are {len(missing_inconsistent_subjects)} subjects with missing values")
        subjects_df = subjects_df.dropna()
        visits_df = filter_visits_by_subjects(subjects_df, visits_df)
        # For SES, we'll just take the first
        subjects_df, visits_df = add_subject_vars(subjects_df, visits_df, ["ses_parent_yoe", "race_label", "hispanic"], remove_from_visits=True, mode="first")
        print(f"After removing inconsistent subject rows, there are {len(subjects_df)} subjects and {len(visits_df)} visits.")
        # Leave only 5 race/ethnicity categories: White/Caucasian, Hispanic, African-American/Black, Asian and Other
        # + There is one individual without a race/ethnicity label (NCANDA_S00649), define as 'Other'
        subjects_df = subjects_df.replace({'race_label': {None: 'Other', 'African-American_Caucasian': 'Other',  'Asian_Pacific_Islander': 'Other',  'Asian_White': 'Other', 'Native American/American Indian': 'Other', 'NativeAmerican_Caucasian': 'Other', 'Pacific Islander': 'Other', 'Pacific_Islander_Caucasian': 'Other'}})
        subjects_df['race_ethnicity'] = subjects_df.apply(lambda row: 'Hispanic' if row['hispanic'] == 'Y' else row['race_label'], axis=1)
        subjects_df = subjects_df.drop("race_label", axis=1)
        subjects_df = subjects_df.drop("hispanic", axis=1)
        var_lst = [x for x in var_lst if x not in ["hispanic", "race_label", "ysr_anxdep_raw", "asr_anxdep_raw", "ysr_withdep_raw", "asr_withdrawn_raw", "ysr_somatic_raw", "asr_somatic_raw", "ysr_thought_raw", "asr_thought_raw", "ysr_attention_raw", "asr_attention_raw", "ysr_aggress_raw", "asr_aggressive_raw", "ysr_rulebrk_raw", "asr_rulebreak_raw"]]
        var_lst.insert(3, "race_ethnicity")
        var_groups["race_ethnicity"], var_subgroups["race_ethnicity"] = "redcap", "demographics"
        subject_vars = ["family_id", "race_ethnicity", "sex", "site", "ses_parent_yoe"]
        # If we are using structural features, average the left and right sides
        # Update the variables dataframe
        variables_df = update_variables(subjects_df, visits_df, pd.DataFrame({"var": var_lst, "subject_var": [v in subject_vars for v in var_lst], "group": [var_groups[v] for v in var_lst], "subgroup": [var_subgroups[v] for v in var_lst]}))
        return subjects_df, visits_df, variables_df

