"""Parent class for loading subjects and visits dataframes from different formats. These are
stored as two Pandas dataframes in the given path.
A subjects_df has, among others, the columns subject_id, sex_at_birth ('F', 'M', 'O'), ethnicity
A visits_df has the columns subject_id, months_after_baseline (0 for baseline), visit_date, site_id
"""

import os
import lln.utils.io as io

class SubjectsVisitsLoader:
    ''''A class to load tabular data from a variaty of datasets'''
    def __init__(self, output_path):
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def get_data(self, df_name, reload=False):
        ''''Fetches the dataframes if available, otherwise reads the data and saves them'''
        try:
            assert not reload
            subjects_df = io.load_df(self.output_path, f"subjects_{df_name}.csv")
            visits_df = io.load_df(self.output_path, f"visits_{df_name}.csv")
            variables_df = io.load_df(self.output_path, f"variables_{df_name}.csv")
            return subjects_df, visits_df, variables_df
        except (FileNotFoundError, AssertionError):
            print(f"Files subjects_{df_name}.csv, visits_{df_name}.csv and variables_{df_name}.csv being read and stored in {self.output_path}")
            subjects_df, visits_df, variables_df = self.read_data()
            io.dump_df(subjects_df, self.output_path, f"subjects_{df_name}.csv")
            io.dump_df(visits_df, self.output_path, f"visits_{df_name}.csv")
            io.dump_df(variables_df, self.output_path, f"variables_{df_name}.csv")
            return subjects_df, visits_df, variables_df
        
    def read_data(self):
        '''Reads data into dataframes, must be defined by subclasses'''
        return None, None, None