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
    X1 = np.random.exponential(3,nobs).reshape(nobs,1)
    X2 = np.random.gamma(3,2,nobs).reshape(nobs,1)
    X0 = np.ones((nobs,1))
    eps = np.matrix(np.random.randn(nobs).reshape(nobs,1))
    beta = np.matrix(np.random.random(3).reshape(3,1))
    #return(np.size(X),np.size(beta))
    X = np.matrix(np.hstack((X0,X1,X2)))
    y = np.dot(np.matrix(X),np.matrix(beta)) + eps
    return(X,y,beta)


def compare_models():
    """
    Compares output from different implementations of OLS.
    INPUT
        X (ndarray) the independent variables in matrix form
        y (array) the response variables vector
    RETURNS
        results (pandas.DataFrame) of estimated beta coefficients
    """
    pass


def load_hospital_data():
    """
    Loads the hospital charges data set found at data.gov.
    INPUT
        path_to_data (str) indicates the filepath to the hospital charge data (csv)
    RETURNS
        clean_df (pandas.DataFrame) containing the cleaned and formatted dataset for regression
    """
    pass


def prepare_data():
    """
    Prepares hospital data for regression (basically turns df into X and y).
    INPUT
        df (pandas.DataFrame) the hospital dataset
    RETURNS
        data (dict) containing X design matrix and y response variable
    """
    pass


def run_hospital_regression():
    """
    Loads hospital charge data and runs OLS on it.
    INPUT
        path_to_data (str) filepath of the csv file
    RETURNS
        results (str) the statsmodels regression output
    """
    pass
 

### END ###