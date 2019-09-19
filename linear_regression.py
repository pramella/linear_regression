### Tools for linear regression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def simulate_data(nobs):
    """
    Simulates data for testing linear_regression models.
    INPUT
        nobs (int) the number of observations in the dataset
    RETURNS
        data (dict) contains X, y, and beta vectors.
    """
    X = np.random.randn(nobs).reshape(nobs,1)
    X0 = np.ones((nobs,1))
    eps = np.matrix(np.random.randn(nobs).reshape(nobs,1))
    beta = np.matrix(np.random.random(2).reshape(2,1))
    #return(np.size(X),np.size(beta))
    X = np.matrix(np.hstack((X0,X)))
    y = np.dot(np.matrix(X),np.matrix(beta)) + beta
    return(X,y,beta)


def compare_models(X, y, beta=None):
    """
    Compares output from different implementations of OLS.
    INPUT
        X (ndarray) the independent variables in matrix form
        y (array) the response variables vector
    RETURNS
        results (pandas.DataFrame) of estimated beta coefficients
    """

    # Using statsmodels OLS
    output_sm = sm.OLS(y, X).fit()
    beta_sm = output_sm.params

    # Using sklearn's Linear Regression
    output_skl = LinearRegression(fit_intercept=False).fit(X, y)
    beta_skl = output_skl.coef_

    results = pd.DataFrame()
    results['statsmodels'] = beta_sm
    results['sklearn'] = beta_skl

    if beta is not None:
        results['truth'] = beta

    return results


def load_hospital_data(path_to_data):
    """
    Loads the hospital charges data set found at data.gov.
    INPUT
        path_to_data (str) indicates the filepath to the hospital charge data (csv)
    RETURNS
        clean_df (pandas.DataFrame) containing the cleaned and formatted dataset for regression
    """

    # load csv with pandas
    df = pd.read_csv(path_to_data)

    # rename/reformat the columns
    rename_dict = dict()
    for c in df.columns:
        rename_dict[c] = c.lower().strip()
    df = df.rename(columns=rename_dict)

    # drop unwanted variables
    vars_to_drop = ['hospital referral region description',
                    'provider street address',
                    'provider name',
                    'provider city',
                    'provider zip code']
    df = df.drop(vars_to_drop, axis=1)

    # match provider id with provider state and condense the dataset
    clean_df = pd.pivot_table(df, columns=['provider id'], aggfunc=sum).T

    states_dict = dict()
    for pid in clean_df.index:
        states_dict[pid] = df[df['provider id']==pid]['provider state'].unique()[0]

    clean_df['provider state'] = clean_df.index
    clean_df['provider state'] = clean_df['provider state'].map(states_dict)
    
    # ensure integers where appropriate
    clean_df['total discharges'] = clean_df['total discharges'].astype(np.int64)


    return clean_df


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


def run_hospital_regression(path_to_data):
    """
    Loads hospital charge data and runs OLS on it.
    INPUT
        path_to_data (str) filepath of the csv file
    RETURNS
        results (str) the statsmodels regression output
    """
    df = load_hospital_data(path_to_data)
    data = prepare_data(df)
    results = sm.OLS(data['y'], data['X']).fit().summary().as_text()

    return results
 

### END ###