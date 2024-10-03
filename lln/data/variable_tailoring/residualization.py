"""Residualize cofounding effects.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

def residualize(df, var, new_var_name=None, covs=[], verbose=False, add_back_dummies=False):
    '''Residualize a variable with respect to a set of covariates.'''
                
    df_with_dummies = df.copy()
    
    # Which exog variables are categorical and which are numerical
    is_categorical = {cov: df_with_dummies[cov].dtype == 'object' for cov in covs}
    cat_covs = [cov for cov in covs if is_categorical[cov]]
    num_covs = [cov for cov in covs if not is_categorical[cov]]
    if verbose:
        print(f"Residualizing {var} with respect to {covs}. Categorical covariates: {cat_covs}. Numerical covariates: {num_covs}.")
        for cov in num_covs:
            correlation = np.corrcoef(df[var].values, df[cov].values)[0, 1]
            print(f"Correlation between {var} and {cov} before residualization: {correlation:.2f}")
        for cov in cat_covs:
            reference_group = df[cov].unique()[0]  # The first category is used as reference (dropped by get_dummies)
            print(f"Reference group for {cov}: {reference_group}")
            encoder = OneHotEncoder(sparse=False)
            encoded_categorical = encoder.fit_transform(df[cov].values.reshape(-1, 1))
            data = np.column_stack((df[var].values, encoded_categorical))
            correlation_matrix = np.corrcoef(data, rowvar=False)
            print(f"Correlation between {var} and {cov} before residualization: {correlation_matrix}")
    if len(cat_covs) > 0:
        #for cov in cat_covs:
        #    reference_group = df[cov].unique()[0]  # The first category is used as reference (dropped by get_dummies)
        #    print(f"Reference group for {cov}: {reference_group}")
        df_with_dummies.rename(columns={col: f'caty_{col}' for col in cat_covs}, inplace=True)
        cat_cov_dummies = [f'caty_{col}' for col in cat_covs]
        # Convert the categorical variables into dummy variables
        #'drop_first=True' drops the first category to avoid multicollinearity
        df_with_dummies = pd.get_dummies(df_with_dummies, columns=cat_cov_dummies, drop_first=True, dtype=int)            
    
    # Exog: The covariates we want to control for
    # Selecting the relevant variables for the independent variables (exog)
    # This includes all numerical variables and the dummy variables for the categorical variables
    exog_columns = num_covs + [col for col in df_with_dummies.columns if 'caty_' in col]
    exog = sm.add_constant(df_with_dummies[exog_columns])

    # Endog: The variable we want to residualize, i.e. use in the analysis
    endog = df_with_dummies[var] 
    
    # Generalized linear model
    model = sm.GLM(endog, exog, family=sm.families.Gaussian()).fit()
    #if verbose:
    #    print(model.summary())

    # Calculate residual variables
    if not new_var_name:
        new_var_name = var
    residuals = model.resid_response
    
    # Add back the dummy variable effects
    if add_back_dummies:
        print("Adding back the dummy variable effects.")
        for col in exog_columns:
            if 'caty_' in col:
                residuals += model.params[col] * df_with_dummies[col]
    
    df.loc[:, new_var_name] = residuals
    
    if verbose:
        for cov in num_covs:
            correlation = np.corrcoef(df[var].values, df[cov].values)[0, 1]
            print(f"Correlation between {var} and {cov} after residualization: {correlation:.2f}")
        for cov in cat_covs:
            encoder = OneHotEncoder(sparse=False)
            encoded_categorical = encoder.fit_transform(df[cov].values.reshape(-1, 1))
            data = np.column_stack((df[var].values, encoded_categorical))
            correlation_matrix = np.corrcoef(data, rowvar=False)
            print(f"Correlation between {var} and {cov} after residualization: {correlation_matrix}")
    return df

def residualize_ignore_nans(df, var, new_var_name=None, covs=[], verbose=False, add_back_dummies=False):
    using_tmp_name = False
    if new_var_name is None or new_var_name == var:
        new_var_name = f"{var}_temp"
        using_tmp_name = True
    df[new_var_name] = np.nan
    df_no_nans = df.dropna(subset=[var]+covs)
    print(f"First removing NaNs from {var} and covs. Size of df: {len(df)}. Size of df_no_nans: {len(df_no_nans)}")
    df_no_nans = residualize(df_no_nans, var, new_var_name, covs=covs, verbose=verbose, add_back_dummies=add_back_dummies)
    df.update(df_no_nans)
    if using_tmp_name:
        df[var] = df[new_var_name]
        df.drop(columns=[new_var_name], inplace=True)
    return df