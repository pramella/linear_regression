import pandas as pd
def prepare_data(df):
    """
    Prepares hospital data for regression (basically turns df into X and y).
    INPUT
        df (pandas.DataFrame) the hospital dataset
    RETURNS
        data (dict) containing X design matrix and y response variable
    """
    data = dict()

    X1 = np.log(df['total discharges'].values)
    X2 = pd.get_dummies(df['provider state'])
    X = np.column_stack((np.ones(len(X1)), X1, X2))

    Y = np.log(df['average covered charges'].values)

    data['X'] = X
    data['y'] = Y

    return data