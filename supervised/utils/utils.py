'''Functions related to outlier calculation and fixing them.'''

def find_outliers(df, column):
    # Test the data for outliers using Interquartile range
    ''' The IQR is the difference between the 25th percentile (Q1) and the 75th percentile (Q3).
    Outliers are typically defined as values that are below Q1 - 1.5 * IQR or above Q3+1.5*IQR
    '''
    # Calculate the IQR for studytime; Q represent the Quantile
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return lower_bound, upper_bound, df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Fix the outliers
def fix_outliers(df, column):
    ''' Fix the outliers by first finding them and capping the values between
    lower and upper bounds. '''
    lower_bound, upper_bound, outliers = find_outliers(df, column)
    if not outliers.empty:
        print(f"Fixing outliers: {len(outliers)}")

    lower_bound = df[column].dtype.type(lower_bound)
    upper_bound = df[column].dtype.type(upper_bound)

    df.loc[df[column] < lower_bound, column] = lower_bound
    df.loc[df[column] > upper_bound, column] = upper_bound

    return df[column]
