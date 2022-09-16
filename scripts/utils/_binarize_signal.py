def _identify_onsets(df, col_name, ):

    """
    Sum up two integers
    Arguments:
        a: an integer
        b: an integer
    Returns:
        The sum of the two integer arguments
    """
    # need pandas datafrrame
    # need column of pandas dataframe
    # need 
    # output:
    # two lists of indices
    # dictionary?
    import numpy as np
    mid_val = (np.max(df['expect']) - np.min(df['expect']))/2
    df.loc[df['expect'] > mid_val, 'expect_rating'] = 5
    df.loc[df['expect'] <= mid_val, 'expect_rating'] = 0