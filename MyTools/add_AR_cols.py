import pandas as pd

def add_AR_cols(df: pd.DataFrame, lags: int, col: str= 'PX_OPEN_VIX_volatility', return_cols=False)-> pd.DataFrame:
    """
    added columns defined by VIX_lagged_i = VIX - VIX_-i
    :param df: DataFrame containing all variables
    :param lags: number of passed unit time added to the DataFrame
    :return: the DataFrame with the lagged columns
    """
    cols=[]

    for i in range(1,lags):
        df['VIX_LAG_' + str(i)] = df[col] - df[col].shift(i)
        cols.append('VIX_LAG_' + str(i))


    if return_cols: return df.dropna(), cols
    else: return df.dropna()
    
