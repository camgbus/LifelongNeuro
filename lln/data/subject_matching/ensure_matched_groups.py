"""Ensure that two groups are properly matched with regards to multiple covariates.

For two groups matched according to multiple covariates, use two-sample t-tests to compare the 
means of the two samples to determine if they are significantly different from each other. If they 
are, we can conclude that they were likely drawn from different populations and are not matched.
"""

from scipy import stats

def ensure_matched_groups(df_1, df_2, covariates):
    '''Ensure that two groups are properly matched with regards to multiple covariates.
    
    Parameters:
        group_1 (numpy.ndarray): First group
        group_2 (numpy.ndarray): Second group
    Returns:
        None
    '''
    # Perform two-sample t-test
    t_statistics, p_values = stats.ttest_ind(df_1[covariates], df_2[covariates])

    # Display results
    print("Results of t-test:")
    print("T-statistic:", t_statistics)
    print("P-value:", p_values)

    # Interpret results
    alpha = 0.05  # Significance level
    for p_value, cov in zip(p_values, covariates):
        if p_value < alpha:
            print(f"Cov {cov}: Null hypothesis rejected: means are significantly different")
        else:
            print(f"Cov {cov}: Null hypothesis cannot be rejected: no significant difference in means")