import os
from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from MyTools.add_AR_cols import add_AR_cols
from data_treatment.box_creator import create_binary_box

if not "root" in locals():
    current_path = Path(os.getcwd())
    root = current_path.parent.absolute()
os.chdir(root)

PATH = "../"

cols_selected_by_lasso = ['PX_OPEN_VIX_volatility', 'VOLUME_TOTAL_CALL_VIX_volatility',
       '3MTH_IMPVOL_110.0%MNY_DF_VIX_volatility',
       'PUT_CALL_VOLUME_RATIO_CUR_DAY_SPX_volatility',
       'VOLUME_TOTAL_PUT_SPX_volatility', 'QMJ USA_QMJ Factors',
       'BAB Global_BAB Factors', 'Bullish_SENTIMENT',
       'Bullish 8-week Mov Avg_SENTIMENT',
       'Mkt-RF_F-F_Research_Data_5_Factors_2x3_daily',
       'SMB_F-F_Research_Data_5_Factors_2x3_daily', 'VIX_LAG_1', 'VIX_LAG_3',
       'VIX_LAG_4', 'VIX_LAG_5']

def easy_data_import():
    df = pd.read_csv('database.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = create_binary_box(df, relative_threshold = 0.05, box_length=5).set_index(['Date']).dropna(axis = 0)
    df = add_AR_cols(df,7)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols_selected_by_lasso])
    y = df['Box']

    ts_cv = TimeSeriesSplit()
    return X, y, ts_cv