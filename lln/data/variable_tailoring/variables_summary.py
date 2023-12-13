"""Generates/updates a rough summary of the variables in a data frame
"""
import pandas as pd
from pandas.api.types import is_numeric_dtype

def update_variables_df(variables_df, data_set):
    variable_tuples = list(variables_df.itertuples(index=False))
    new_variables = []
    for var, group, subgroup, _, _, derived_from in variable_tuples:
        try:
            values = data_set[var]
            is_continual = is_numeric_dtype(values)
            if is_continual:
                possible_values = (min(values), max(values))
            else:
                possible_values = sorted(list(values.dropna().unique()))
            new_variables.append([var, group, subgroup, is_continual, possible_values, derived_from])
        except KeyError:
            print(f"Variable {var} no longer in the data")
    variables_cols = ["var", "group", "subgroup", "continuous", "range", "derived_from"]
    return pd.DataFrame(new_variables, columns=variables_cols)