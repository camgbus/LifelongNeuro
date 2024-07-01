"""Residualize cofounding effects.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

def residualize(df, var, new_var_name=None, covs=[], verbose=False):
    '''Residualize a variable with respect to a set of covariates.'''
                
    df_with_dummies = df.copy()
    
    # Which exog variables are categorical and which are numerical
    is_categorical = {cov: df_with_dummies[cov].dtype == 'object' for cov in covs}
    cat_covs = [cov for cov in covs if is_categorical[cov]]
    num_covs = [cov for cov in covs if not is_categorical[cov]]
    if verbose:
        correlation = np.corrcoef(df[var].values, df[num_covs[0]].values)[0, 1]
        print(f"Residualizing {var} with respect to {covs}. Categorical covariates: {cat_covs}. Numerical covariates: {num_covs}. Correlation between {var} and {num_covs[0]} before residualization: {correlation:.2f}")
    if len(cat_covs) > 0:
        df_with_dummies.rename(columns={col: f'caty_{col}' for col in cat_covs}, inplace=True)
        cat_covs = [f'caty_{col}' for col in cat_covs]
        # Convert the categorical variables into dummy variables
        #'drop_first=True' drops the first category to avoid multicollinearity
        df_with_dummies = pd.get_dummies(df_with_dummies, columns=cat_covs, drop_first=True, dtype=int)            
    
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
    df[new_var_name] = residuals
    
    if verbose:
        correlation = np.corrcoef(residuals, df[num_covs[0]].values)[0, 1]
        print(f"Correlation between {var} and {num_covs[0]} after residualization: {correlation:.2f}")
        
    return df