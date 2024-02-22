"""Residualize cofounding effects.
"""
import statsmodels.api as sm

def residualize(df, var, new_var_name=None, covs=[], verbose=False):
    '''Redidualize a variable with respect to a set of covariates.'''
                
    if not new_var_name:
        new_var_name = var
            
    # Exog: The covariates
    exog = sm.add_constant(df[covs])
            
    # Endog: The variable we want to residualize, i.e. use in the analysis
    endog = df[var] 
    
    # Generalized linear model
    model = sm.GLM(endog, exog, family=sm.families.Gaussian()).fit()
    if verbose:
        print(model.summary())

    # Calculate residual variables
    df[new_var_name] = model.resid_response
    
    return df