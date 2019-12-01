import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, Normalizer, PolynomialFeatures

def validate_rows(dataFrame, row, col_label):
    """
    Validate each row by ensuring assumption that val(x1) < val(x2) < val(x3) 
    always holds. Should it be broken, swap row approriately the rows.
    :param dataFrame: pandas DataFrame storing data collected by a probe
    :param row: A given row (sample) to sort
    :param col_label: A letter indicating a component (each of 3 resolutions) to sort
    :return: DataFrame with correctly assigned specified row 
    """
    df = dataFrame.copy()

    # sort columns into correct order locally
    vs = [df[col_label+'1'][row], df[col_label+'2'][row], df[col_label+'3'][row]]
    vs.sort()

    # fix potentially incorrect ordeting in the DataFrame
    df[col_label+'1'][row] = vs[0]
    df[col_label+'2'][row] = vs[1]
    df[col_label+'3'][row] = vs[2]

    return df       

def preprocess_probe_data(dataFrame):
    """
    Remove outliers occurring due to swaps in column values for some samples (rows).
    :param dataFrame: pandas DataFrame storing data collected by a probe
    :return: Return processed probeA samples in a DataFrame
    """
    df = dataFrame.copy()
    
    # rows validated seprately to allow easier tuning and debugging
    for row in range(0, df.index.size):
        df = validate_rows(df, row, 'c')
        df = validate_rows(df, row, 'm')
        df = validate_rows(df, row, 'n')
        df = validate_rows(df, row, 'p')
    
    return df

def std_norm(df_train, df_test, std=True, norm=True):
    """
    Standardise and normalises training and testing data (features).
    :param df_train: DataFrame with training frature data
    :param df_test: DataFrame with test feature data
    :param std: if True, data will be standardised (default False)
    :param nrom: if True, data will be normlaised (default False)
    :return: A pair of transformed dataFrames
    """
    x_train, x_test = df_train.copy(), df_test.copy()

    if std == True:
        standarizer = StandardScaler().fit(x_train)
        x_train  = standarizer.transform(x_train)
        standarizer = StandardScaler().fit(x_test)
        x_test = standarizer.transform(x_test)

    if norm == True:
        normalizer = Normalizer().fit(x_train)
        x_train  = normalizer.transform(x_train)
        normalizer = Normalizer().fit(x_test)
        x_test = normalizer.transform(x_test)

    return x_train, x_test

def predict_tna(df_xa, df_xb, df_ta):
    """
    Given training data {df_xa,df_ta} predict 'tna' for samples df_xb.
    :param df_xa: training samples 
    :param df_xb: samples to predict 'tna' for
    :param df_ta: training 'tna' targets for df_xa
    :return: DataFrame containing 'tna' predicitons
    """
    x_a, x_b, t_a = df_xa.copy(), df_xb.copy(), df_ta.copy()

    # Standardise feature data x
    x_a, x_b = std_norm(x_a, x_b, norm=False)

    # Perform polynomial expansion, degree found using hyperparam tuning
    x_a = PolynomialFeatures(3).fit_transform(x_a)
    x_b = PolynomialFeatures(3).fit_transform(x_b)

    # Fit data to the model, alpha found using LassoCV
    lasso = linear_model.Lasso(alpha=0.012)
    lasso.fit(x_a, t_a)
    t_pred = pd.DataFrame(lasso.predict(x_b), columns=['tna'])

    return pd.DataFrame(t_pred, columns=['tna'])