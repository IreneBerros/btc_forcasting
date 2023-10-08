import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from numba import jit
import seaborn as sns
from datetime import datetime, timedelta

from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#own functions -> utils_model.py
import utils_model as um

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from decimal import Decimal



#--------------------------------IMPORT DATA-------------------------------------
df_off = pd.read_csv('../model/data/off_chain.csv', sep=';', skiprows=1)
df_off['time'] = df_off['time'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%m-%d-%Y'))
df_off['time'] = pd.to_datetime(df_off['time'])

df_off2 = pd.read_csv('../model/data/off_chain_clean.csv')
df_off2.columns = ['time', 'ratio_mvrv', 'ratio_met', 'ratio_nvt', 'ratio_stf']
df_off2['time'] = pd.to_datetime(df_off2['time'])

df_off3 = pd.read_csv('../model/data/off_chain_monthly.csv')
df_off3['time'] = pd.to_datetime(df_off3['time'])

df_on = pd.read_csv('../model/data/on_chain_roy.csv', sep=';')
#change Date column to 'time' for future join
df_on['time'] = pd.to_datetime(df_on['Date'])
df_on.drop('Date', axis=1, inplace=True)
df_on['US_bank_idx'].fillna(method='ffill', inplace=True)
df_on['US_bank_idx'][0:3] = df_on['US_bank_idx'][3]

df_on2 = pd.read_csv('../model/data/on_chain_paper.csv')
#change Date column to 'time' for future join
df_on2['time'] = pd.to_datetime(df_on2['timestamp'])
df_on2.drop('timestamp', axis=1, inplace=True)

join1 = df_off.merge(df_off2, on='time', how='left')
join2 = join1.merge(df_off3, on='time', how='left')
join_on_off = join2.merge(df_on, on='time', how='left')
join_on_off2 = join_on_off.merge(df_on2, on='time', how='left')

#Final dataset all data joined
df = join_on_off2.copy()




#-------------------------------------EDA----------------------------------
df.columns = ['time', 'open', 'high', 'low', 'close', 'Upper Bollinger',
       'Lower Bollinger', 'Upper Donchian', 'Lower Donchian', 'Volume',
       'Red Arrow', 'Green Arrow', 'RSI', 'Put-call Ratio',
       'Detrended Price Oscillator', 'ratio_mvrv', 'ratio_met', 'ratio_nvt',
       'ratio_stf', 'Value_M3', 'Value_CPI', 'Value_EFF', 'Value_SP500',
       'Value_USpce', 'Value_USreal', 'Variation_UStotal', 'Value_USunem',
       'hash_rate', 'active_addre', 'tran_sec', 'tran_num', 'block_size',
       'rev_all_miners', 'block_reward', 'size_tran', 'trans_fees',
       'US_bank_idx', 'thermocap-usd', 'price-usd',
       'block-height', 'supply-btc', 'destroyed-cvdd', 'delta-cap-usd',
       'capitalization', 'market-cap-usd', 'ratio-mvrv', 'created-utxos-btc',
       'spent-utxos-btc', 'mining-difficulty', 'z-score', 'balance-0-1',
       'balance-0-01', 'balance-1', 'balance-1k', 'balance-10', 'balance-10k',
       'balance-100', 'number-of-addresses', 'number-of-utxos',
       'percent-of-supply', 'price-drawdown', 'cap-usd', 'realized-price-usd',
       'unrealized-profit', 'stock-to-flow', 'active-3y-5y-btc',
       'active-5y-7y-btc']


date_feat = 'time'
numerical_feat = df.drop(['time','Red Arrow', 'Green Arrow'], axis=1).columns
categorical_feat = ['Red Arrow', 'Green Arrow']

target = 'vol_future'

#deleting the two category columns because they are too unbalanced
df_nonan = df.drop(['Red Arrow', 'Green Arrow'], axis=1)
df_nonan.drop('ratio_met', axis=1, inplace=True)

# columns: Value_M3, _CPI, _EFF, _SP500, _USpce 
# nans from 2023-03-02 to 2023-04-26 -> drop all rows --> new final date 2023-03-01
df_nonan = df_nonan.iloc[:2558]

# column: Value_USreal --> too many nans, drop column, ask Roy
df_nonan.drop('Value_USreal', axis=1, inplace=True)





#--------------------------------FEATURE ENGINEERING--------------------------------

df_nonan['time'] = pd.to_datetime(df_nonan['time'])
df_nonan.set_index('time', inplace=True)

df_nonan['log_price'] = np.log(df_nonan['close'].astype(float))
df_nonan['returns'] = df_nonan['close'].astype(float).pct_change().dropna() *100
df_nonan['log_return'] = df_nonan['log_price'] - df_nonan['log_price'].shift(1)

df_nonan['HL_sprd'] = np.log((df_nonan.high - df_nonan.low) / df_nonan.close)
df_nonan['CO_sprd'] = (df_nonan.close - df_nonan.open) / df_nonan.open
df_nonan['Volume'] = np.log(df_nonan.Volume) 

# DROPPING THE 1ST ROW OF DATA 
# BECAUSE I SHIFTED IT FORWARD TO CALCULATE RETURNS/LOG RETURNS
df_nonan.drop(['open', 'high', 'low', 'close', 'Lower Bollinger',
       'Upper Donchian', 'Lower Donchian'], axis=1, inplace=True)




#----VOLATILITY-----

intervals = [7, 30, 60, 180, 365]
vols_df = {}

# ITERATE OVER intervals LIST
for i in intervals:
    # GET DAILY LOG RETURNS USING THAT INTERVAL
    vols = df_nonan.log_return.rolling(window=i)\
                         .apply(um.realized_volatility_daily).values

    vols_df[i] = vols

# CONVERT vols_df FROM DICTIONARY TO PANDAS DATAFRAME
vols_df = pd.DataFrame(vols_df, columns=intervals, index=df_nonan.index)
vols = df_nonan.log_return.rolling(window=i)


INTERVAL_WINDOW = 30
n_future = 7


# GET BACKWARD LOOKING REALIZED VOLATILITY
df_nonan['vol_current'] = df_nonan.log_return.rolling(window=INTERVAL_WINDOW)\
                                   .apply(um.realized_volatility_daily)

# GET FORWARD LOOKING REALIZED VOLATILITY 
df_nonan['vol_future'] = df_nonan.log_return.shift(-n_future)\
                                 .rolling(window=INTERVAL_WINDOW)\
                                 .apply(um.realized_volatility_daily)

# DROPPING ALL NaN VALUES
df_nonan.dropna(inplace=True)





#--------------------------------MULTIVARIATE ANALYSIS-----------------------------------------

df_nonan.drop(['returns'], axis=1, inplace=True)

df_nonan_corr = df_nonan.drop(['Upper Bollinger', 'ratio_stf', 'Value_USpce', 'hash_rate', 'active_addre', 'tran_sec',
               'size_tran', 'supply-btc', 'thermocap-usd', 'price-usd', 'block-height', 'destroyed-cvdd',
               'delta-cap-usd', 'capitalization', 'market-cap-usd', 'ratio-mvrv', 'created-utxos-btc', 'mining-difficulty', 
               'z-score', 'balance-0-1', 'balance-0-01', 'balance-1', 'number-of-addresses', 'balance-1k', 'number-of-utxos',
               'active-3y-5y-btc', 'price-drawdown', 'cap-usd','realized-price-usd'], axis=1)

df_final_corr = df_nonan_corr.drop(['stock-to-flow', 'active-5y-7y-btc', 'spent-utxos-btc'], axis=1)




#--------------------------------VOLATILITY-----------------------------------------

df = df_final_corr.copy()
#update numerical & target features

target = ['vol_future']
#numerical_feat = df.drop(['vol_future'], axis=1).columns
numerical_feat = df.drop(['vol_future'], axis=1).columns

# validation/test splits
test_size = 30
val_size = 365

# to index
split_time_1 = len(df) - (val_size + test_size)
split_time_2 = len(df) - test_size

train_idx = df.index[:split_time_1]
val_idx = df.index[split_time_1:split_time_2]
test_idx = df.index[split_time_2:]

# Y (target) 3 splits
y_train = df.loc[train_idx][target]
y_val = df.loc[val_idx][target]
y_test = df.loc[test_idx][target]

# X 3 splits
x_train = df.loc[train_idx].drop('vol_future', axis=1)
x_val = df.loc[val_idx].drop('vol_future', axis=1)
x_test = df.loc[test_idx].drop('vol_future', axis=1)




#--------------------------------LSTM-----------------------------------------

input_df = df[['log_return', 'block_reward', 'CO_sprd', 'vol_current']]
#input_df = df[['log_return', 'trans_fees', 'Volume', 'HL_sprd']]


# CREATE DATASET THAT COMBINES BOTH TRAINING & VALIDATION
#tv_df = df[['log_return', 'trans_fees', 'Volume', 'HL_sprd']][:split_time_2]
tv_df = input_df[:split_time_2]
tv_y = df.vol_future[:split_time_2]


tf.keras.backend.clear_session()

# SET SEED FOR REPRODUCIBILITY
np.random.seed(1234)

n_past = 30
batch_size = 16
n_dims = input_df.shape[1]

mat_X_tv, mat_y_tv = um.windowed_dataset(tv_df, tv_y, n_past)

# CONSTRUCTING MULTIVARIATE BIDIRECTIONAL LSTM NN
lstm_final = tf.keras.models.Sequential([  
    tf.keras.layers.InputLayer(input_shape=[n_past, n_dims]),   
    # BATCH NORMALIZATION  
    tf.keras.layers.BatchNormalization(), 

    # ADDING 1st LSTM LAYER
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    
    # ADDING 2nd LSTM LAYER
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.2),

    # DENSE OUTPUT LAYER
    tf.keras.layers.Dense(1)
])

lstm_final.compile(loss='mse', 
                    optimizer="adam", 
                    metrics=[um.rmspe])

checkpoint_cb = ModelCheckpoint('lstm_final.h5',
                                save_best_only=True,
                                monitor='val_rmspe')

# STOPPING THE TRAINING IF VALIDATION RMSPE IS NOT IMPROVING 
early_stopping_cb = EarlyStopping(patience=30,
                                  restore_best_weights=True,
                                  monitor='val_rmspe')

print(lstm_final.summary())

lstm_final_res = lstm_final.fit(mat_X_tv, mat_y_tv, epochs=500,
                                validation_split=0.2,
                                callbacks=[checkpoint_cb, early_stopping_cb],
                                verbose=0, batch_size=batch_size, shuffle=True)



# FORECASTING ON VALIDATION SET
y_test_preds = um.forecast_multi(lstm_final, test_idx, input_df=input_df, df=df, n_past=n_past)

print('RMSPE on Test Set:', um.RMSPE(y_test['vol_future'], y_test_preds))

y_test_preds.to_csv('predictions.csv')


