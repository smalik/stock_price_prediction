# Questions of interest:
#   1. How do we define performance for price prediction.
#       a. How accurate does it need to be? 80% precision.
#           i. Entry is the open of the day.  Take the position on a daily period based on the opening prediction with respect to the close price.
#       b. What does accuracy mean? See above
#       c. What is the metric for accuracy? Avg. daily loss
#       d. What is a suitable window for building a model (price action is temporally scoped)?
#           i. investing vs. trading horizons?
#       e. Experiment:
#           i. Take 500 tickers and track direction and also magnitude
#   2. How many periods out do we need to be predicting price?
#   3. How do we think about the indicators on the price action?
#       a. Is it more appropriate to think of the indicators as a policy on the predition?
#       b. Do we think of creating dynamic rules (i.e. decision trees) of indicators over predicted price action)?
#   4. Choice of modeling method matters.
#       a. SVM resulted in overfitting.  What is the bias/variance tradeoff?
#       b. Hard to think about indicators outside of ARIMA time series regression.
#           i. ARIMA time series regression is very compute intensive
#           ii. Not as much predictive performance as neural networks.  Neural Network are tough to tune, but a good model need to be updates less regularly than the ARIMA
#               i. https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
#       c. Does candlestick pattern recognition matter for this analysis objective?
#           i. Would it help to match the current pattern with prior pattern and time slice from history?
#           ii. What are optimal sliding windows for learning?  What price-time aperture makes sense to learn?
#   5. Things I am working on:
#       a. Hyperparameter tuning.  Dont have a good sense of what a sufficient level of bias/variance tradeof is
#       b. Deciding on the right kind of loss leasure
#       c. Incorporate policy controls for entry-exit strategies
#       d. Combining ARIMA with LTSM and seq2seq
#       e. Backtesting harness
#       f. Methodology to partition price histories into segmented models for later recombination.  Basically an ensemble approach.
#       g. What is the right way to think about the relationship of other indicies and macroeconomic data to the price predictions?
#       h. Covarying indexes
#           a. SPY500
#           b. 5 Year bond price index (is a leading predictor, but window not stable)
#           c. 5 Year treasury rate
#           d. VIX & other volatility indicators
#           e.


import os
import numpy as np
import time
import random
import ssl
import pandas as pd
import pandas_ta as ta
from iexfinance.stocks import Stock, get_historical_data, get_historical_intraday
from datetime import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from yahoo_fin import stock_info as si
from collections import deque
import pandas_ml as pml

API_TOKEN = 'pk_dc6d30da3c194003b73023caa63d99a8'

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
ssl._create_default_https_context = ssl._create_unverified_context

# get ticker data from yahoo finance
# https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras
rich_features = ['open', 'high', 'low', 'close', 'adjclose', 'volume',
                 'rsi', 'macd', 'adx_14', 'DMP_14', 'DMN_14', 'macdh', 'macds', 'tsi', 'mom',
                 'hl2', 'hlc3',  'ohlc4', 'midpoint', 'midprice',
                 'tema', 'wma10', 'wma20', 'wma30', 'wma50', 'wma200',
                 'sma10', 'sma20', 'sma30', 'sma50', 'sma200', 'ema50', 'ohlc4',
                 'log_return', 'percent_return', 'stdev', 'zscore', 'quantile',
                 'mad', 'adx14', 'eom', 'pvol', 'efi', 'pvt', 'cci', 'increasing',
                 'decreasing', 'bolinger_bands_l20', 'bolinger_bands_m20', 'bolinger_bands_u20',
                 'roc_10', 'stoch_14', 'vwap',
                 'cross_sma10', 'cross_ema50', 'cross_sma200', 'cross_macd']

# fetch data from yahoo finance
def fetch_ticker_data(ticker) -> pd.DataFrame:
    if isinstance(ticker, str):
        _df = si.get_data(ticker)
        return _df

def load_data(ticker_data,
              n_steps=50,
              scale=True,
              shuffle=True,
              lookup_step=1,
              test_size=0.2,
              feature_columns: list=['adjclose', 'volume', 'open', 'high', 'low'],
              nan_offset: int = 50,
              ):
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker_data, pd.DataFrame):
        # load it from yahoo_fin library
        _df = ticker_data
        #_df['neg_volume'] = _df.ta.vp().iloc[:, 4]
        #_df['pos_volume'] = _df.ta.vp().iloc[:, 3]
        _df['rsi'] = _df.ta.rsi()
        _df['macd'] = _df.ta.macd().iloc[:, 0]
        _df['adx_14'] = _df.ta.adx().iloc[:,0]
        _df['DMP_14'] = _df.ta.adx().iloc[:,1]
        _df['DMN_14'] = _df.ta.adx().iloc[:,2]
        _df['macdh'] = _df.ta.macd().iloc[:, 1]
        _df['macds'] = _df.ta.macd().iloc[:, 2]
        _df['tsi'] = _df.ta.tsi()
        _df['mom'] = _df.ta.mom()
        _df['hl2'] = _df.ta.hl2()
        _df['hlc3'] = _df.ta.hlc3()
        _df['midpoint'] = _df.ta.midpoint()
        _df['midprice'] = _df.ta.midprice()
        _df['tema'] = _df.ta.tema()
        _df['wma10'] = _df.ta.wma()
        _df['wma20'] = _df.ta.wma(length= 20, append=True)
        _df['wma30'] = _df.ta.wma(length= 30, append=True)
        _df['wma50'] = _df.ta.wma(length= 50, append=True)
        _df['wma200'] = _df.ta.wma(length= 200, append=True)
        _df['sma10'] = _df.ta.sma()
        _df['sma20'] = _df.ta.sma(length= 20, append=True)
        _df['sma30'] = _df.ta.sma(length= 30, append=True)
        _df['sma50'] = _df.ta.sma(length= 50, append=True)
        _df['sma200'] = _df.ta.sma(length= 200, append=True)
        _df['ema50'] = _df.ta.ema(length= 50, append=True)
        _df['ohlc4'] = _df.ta.ohlc4()
        _df['log_return'] = _df.ta.log_return()
        _df['percent_return'] = _df.ta.percent_return()
        _df['stdev'] = _df.ta.stdev()
        _df['zscore'] = _df.ta.zscore()
        _df['quantile'] = _df.ta.quantile()
        _df['mad'] = _df.ta.mad()
        _df['adx14'] = _df.ta.adx().iloc[:, 0]
        _df['eom'] = _df.ta.eom()
        _df['pvol'] = _df.ta.pvol()
        _df['efi'] = _df.ta.efi()
        _df['pvt'] = _df.ta.pvt()
        _df['increasing'] = _df.ta.increasing()
        _df['decreasing'] = _df.ta.decreasing()
        #_df['long_run'] = _df.ta.long_run()
        _df['bolinger_bands_l20'] = _df.ta.bbands().iloc[:,0]
        _df['bolinger_bands_m20'] = _df.ta.bbands().iloc[:,1]
        _df['bolinger_bands_u20'] = _df.ta.bbands().iloc[:,2]
        _df['roc_10'] = _df.ta.roc()
        _df['stoch_14'] = _df.ta.stoch().iloc[:,0]
        #_df['chopiness_index'] = _df.ta.chop()
        #_df['psar'] = _df.ta.psar()
        _df['cci'] = _df.ta.cci()
        _df['cross_sma10'] = _df.ta.cross('close', 'sma10')
        _df['cross_ema50'] = _df.ta.cross('close', 'ema50')
        _df['cross_sma200'] = _df.ta.cross('close', 'sma200')
        _df['cross_macd'] = _df.ta.cross('macd', '')
        _df['vwap'] = _df.ta.vwap()
        df = _df.iloc[nan_offset:, :].dropna(axis=1).copy()

    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    print(df.columns)
    result['df'] = df.copy()
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        print(col)
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
    # this last_sequence will be used to predict in future dates that are not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    # split the dataset
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    # return the result
    return result


table=pd.read_csv('~plytos/trading/nasdaq-listed-symbols.csv')
df = table.Symbol
#df.to_csv('S&P500-Info.csv')
#df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])
#sp_company_info = getCompanyInfo(sp["Symbol"][:5].tolist())

company_info_to_df = []
for company in df[:50]:
    company_info_to_df.append(sp_company_info[company])

# Get price history for ticker
start = datetime(2020, 1, 1)
end = datetime(2020, 4, 6)
tqqq_stock_history = getHistoricalPrices('TQQQ')
tqqq_intraday = get_historical_intraday('TQQQ', token= API_TOKEN, output_format='pandas')

# # create these folders if they does not exist


N_STEPS = 100                            # Window size or the sequence length
LOOKUP_STEP = 1                         # Lookup step, 1 is the next day
TEST_SIZE = 0.4                         # test ratio size, 0.2 is 20%
FEATURE_COLUMNS = rich_features         # features to use
date_now = time.strftime("%Y-%m-%d")
### model parameters
N_LAYERS = 3
CELL = LSTM                             # LSTM cell
UNITS =   256                           # 256 LSTM neurons
DROPOUT = 0.4                           # 40% dropout
BIDIRECTIONAL = True                   # whether to use bidirectional RNNs
### training parameters
# mean absolute error loss
# LOSS = "mae"
LOSS = "huber_loss"                     # huber loss
OPTIMIZER = "adam"
BATCH_SIZE = 64                          # number of samples fed in at one time.  BATCH_SIZE=1 is online analysis
EPOCHS = 100
ticker = "NIO"
#ticker = ""

ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"

ticker_data = fetch_ticker_data(ticker)
djia_data =  fetch_ticker_data('DJIA')
sp500_data = fetch_ticker_data('SPY')

data = load_data(ticker_data.copy(), N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS, nan_offset= 201)
data["df"].to_csv(ticker_data_filename)
_data = deepcopy(data)

# construct the model
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)
model.save(os.path.join("results", model_name) + ".h5")

# evaluate the model
mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
print("Mean Absolute Error:", mean_absolute_error)
# predict the future price
future_price = predict(model, data)
print(f"Future price after {LOOKUP_STEP} days is ${future_price:.2f} but actual price was {ticker_data.iloc[-1:, 3]}")
#plot_graph(model, data)
print(f'{LOOKUP_STEP}: ', "Accuracy Score:", get_accuracy(model, data))

#   Candidate hyperparameters:
##  Candidate 1:
##    Mean Absolute Error: 2.922829
##    Future price after 1 days is 82.17$
##    1:  Accuracy Score: 0.9661190965092402
#     N_STEPS = 100   #Lookup step, 1 is the next day
#     LOOKUP_STEP = 1   #test ratio size, 0.2 is 20%
#     TEST_SIZE = 0.4     # features to use
#     FEATURE_COLUMNS = rich_features #["adjclose", "volume", "open", "high", "low"]
#     date_now = time.strftime("%Y-%m-%d")
#     ### model parameters
#     N_LAYERS = 3
#     # LSTM cell
#     CELL = LSTM
#     # 256 LSTM neurons
#     UNITS = 256
#     # 40% dropout
#     DROPOUT = 0.4
#     # whether to use bidirectional RNNs
#     BIDIRECTIONAL = False
#     ### training parameters
#     LOSS = "huber_loss"
#     OPTIMIZER = "adam"
#     BATCH_SIZE = 64
#     EPOCHS = 100
#     # Apple stock market
#     ticker = "TQQQ"
##  Candidate 2:
##    Mean Absolute Error: 4.325587
##    Future price after 1 days is 79.14$
##    1:     Accuracy Score: 0.9260780287474333
#     N_LAYERS = 5
##  Candidate 3:
##    Mean Absolute Error: 3.8370905
##    Future price after 1 days is 69.59$
##    1:     Accuracy Score: 0.9260780287474333
##    N_STEPS = 21
##    N_LAYERS = 5
##    BATCH_SIZE = 5
##  Candidate 4:
##      Mean Absolute Error: 3.3179967
##      Future price after 1 days is $77.48 but actual price was 79.22000122070312
##      1:  Accuracy Score: 0.9770240700218819
#     N_STEPS = 100                            # Window size or the sequence length
#     LOOKUP_STEP = 1                         # Lookup step, 1 is the next day
#     TEST_SIZE = 0.4                         # test ratio size, 0.2 is 20%
#     N_LAYERS = 5
#     CELL = LSTM                             # LSTM cell
#     UNITS =   96                           # 256 LSTM neurons
#     DROPOUT = 0.6                           # 40% dropout
#     BIDIRECTIONAL = False                   # whether to use bidirectional RNNs
#     LOSS = "huber_loss"                     # huber loss
#     OPTIMIZER = "adam"
#     BATCH_SIZE = 64                          # number of samples fed in at one time.  BATCH_SIZE=1 is online analysis
#     EPOCHS = 500


# Lets try a classification model

# RF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
df = deepcopy(data['df'])
df['pct_chg'] = df.adjclose.pct_change()
X = df.drop(['ticker'], axis=1).iloc[1:, :]
y = df.pct_chg[1:] > 0.01
#y = df.pct_chg.shift(-1)
y = df.pct_chg
X_train, X_test, y_train, y_test = train_test_split(X, y[:-1], test_size=0.02, random_state=42)

y = df.adjclose
X_train, X_test, y_train, y_test = train_test_split(X.drop(['close'], axis=1), y, test_size=0.02, random_state=42)

X_train = X.drop(['close'], axis=1).iloc[:-30,:]
X_test = X.drop(['close'], axis=1).iloc[-30:,:]
y_train = y[:-31]
y_test = y[-30:]

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)

pred_ = pd.Series(rf.predict(y_test.values.reshape(-1,1)))

pred = pd.Series(rf.predict(y_test.values.reshape(-1,1)))

pred_.set_axis(y_test.axes, inplace=True)
errors_ = pd.DataFrame(deepcopy(y_test))
errors_['preds'] = pred_.values
confusion_matrix = pd.crosstab(errors_['pct_chg'], errors_['preds'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import lag_plot


train_data, test_data = df[0:int(len(df)*0.9)], df[int(len(df)*0.9):]
plt.figure(figsize=(12,7))
plt.title('TQQQ Prices')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.plot(df['close'], 'blue', label='Training Data')
plt.plot(test_data['open'], 'green', label='Testing Data')
plt.xticks(np.arange(0,7982, 1300), df.index[0:7982:1300])
plt.legend()
plt.show()


test_data.close.plot()
plt.show()

def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) +       np.abs(y_true))))

train_ar = X_train['adjclose'].values
test_ar = X_test['adjclose'].values
history = [x for x in train_ar]
print(type(history))
predictions = list()
for t in range(len(test_ar)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)
error = mean_squared_error(test_ar, predictions)
print('Testing Mean Squared Error: %.3f' % error)
error2 = smape_kun(test_ar, predictions)
print('Symmetric mean absolute percentage error: %.3f' % error2)


fig, axes = plt.subplots(3, 2, figsize=(12, 16))
plt.title(f'{ticker} Autocorrelation plot')

# The axis coordinates for the plots
ax_idcs = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1)
]

for lag, ax_coords in enumerate(ax_idcs, 1):
    ax_row, ax_col = ax_coords
    axis = axes[ax_row][ax_col]
    lag_plot(df['close'], lag=lag, ax=axis)
    axis.set_title(f"Lag={lag}")

plt.show()

import pmdarima as pm
from pmdarima.arima import ndiffs

kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)

print(f"Estimated differencing term: {n_diffs}")

auto = pm.auto_arima(y_train, d=n_diffs, seasonal=False, stepwise=True,
                     suppress_warnings=True, error_action="ignore", max_p=6,
                     max_order=None, trace=True)

from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape

def forecast_one_step():
    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0])

model = auto
forecasts = []
confidence_intervals = []

for new_ob in y_test:
    fc, conf = forecast_one_step()
    forecasts.append(fc)
    confidence_intervals.append(conf)

    # Updates the existing model with a small number of MLE steps
    model.update(new_ob)

print(f"Mean squared error: {mean_squared_error(y_test, forecasts)}")
print(f"SMAPE: {smape(y_test, forecasts)}")

fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# --------------------- Actual vs. Predicted --------------------------
axes[0].plot(y, color='blue', label='Training Data')
axes[0].plot(test_data.index, forecasts, color='green', marker='o',
             label='Predicted Price')

axes[0].plot(test_data.index, y_test, color='red', label='Actual Price')
axes[0].set_title('Microsoft Prices Prediction')
axes[0].set_xlabel('Dates')
axes[0].set_ylabel('Prices')

#axes[0].set_xticks(np.arange(0, 7982, 1300).tolist(), df['Date'][0:7982:1300].tolist())
axes[0].legend()


# ------------------ Predicted with confidence intervals ----------------
axes[1].plot(y_train, color='blue', label='Training Data')
axes[1].plot(test_data.index, forecasts, color='green',
             label='Predicted Price')

axes[1].set_title('Prices Predictions & Confidence Intervals')
axes[1].set_xlabel('Dates')
axes[1].set_ylabel('Prices')

conf_int = np.asarray(confidence_intervals)
axes[1].fill_between(test_data.index,
                     conf_int[:, 0], conf_int[:, 1],
                     alpha=0.9, color='orange',
                     label="Confidence Intervals")

axes[1].set_xticks(np.arange(0, 7982, 1300).tolist(), df['Date'][0:7982:1300].tolist())
axes[1].legend()

df.pct_chg[-30:].plot()
plt.show()























#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = df.drop('pct_chg',axis=1).sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data.index[i]
    new_data['Close'][i] = data['close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[-60:,:]
valid = dataset[60:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)






import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd
import pandas_datareader.data as web
import datetime
import numpy as np
from matplotlib import style

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

style.use('ggplot')


# get 2014-2018 data to train our model
start = datetime.datetime(2010,1,1)
end = datetime.datetime(2020,5,1)
df = web.DataReader(ticker, 'yahoo', start, end)

# get 2020 data to test our model on
start = datetime.datetime(2019,5,2)
end = datetime.date.today()
test_df = web.DataReader(ticker, 'yahoo', start, end)

# sort by date
df = df.sort_values('Date')
test_df = test_df.sort_values('Date')

# fix the date
df.reset_index(inplace=True)
df.set_index("Date", inplace=True)
test_df.reset_index(inplace=True)
test_df.set_index("Date", inplace=True)

df.tail()

# Visualize the training stock data:
import matplotlib.pyplot as plt

plt.figure(figsize = (12,6))
plt.plot(df["Adj Close"])
plt.xlabel('Date',fontsize=15)
plt.ylabel('Adjusted Close Price',fontsize=15)
plt.show()


# Rolling mean
close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

plt.figure(figsize = (12,6))
close_px.plot(label='ZS')
mavg.plot(label='mavg')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

import matplotlib.dates as mdates

# change the dates into ints for training
dates_df = df.copy()
dates_df = dates_df.reset_index()

# Store the original dates for plotting the predicitons
org_dates = dates_df['Date']

# convert to ints
dates_df['Date'] = dates_df['Date'].map(mdates.date2num)
dates_df.tail()

# Use sklearn support vector regression to predicit our data:
from sklearn.svm import SVR

dates = dates_df['Date'].values
prices = df['Adj Close'].values

#Convert to 1d Vector
dates = np.reshape(dates, (len(dates), 1))
prices = np.reshape(prices, (len(prices), 1))

svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
svr_rbf.fit(dates, prices)

plt.figure(figsize = (12,6))
plt.plot(dates, prices, color= 'black', label= 'Data')
plt.plot(org_dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# Create train set of adj close prices data:
train_data = df.loc[:,'Adj Close'].values
print(train_data.shape) # 1258

# Apply normalization before feeding to LSTM using sklearn:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)

scaler.fit(train_data)
train_data = scaler.transform(train_data)

'''Function to create a dataset to feed into an LSTM'''
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# Create the data to train our model on:
time_steps = 20
X_train, y_train = create_dataset(train_data, time_steps)

# reshape it [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], time_steps, 1))

print(X_train.shape)

# Visualizing our data with prints:
print('X_train:')
print(str(scaler.inverse_transform(X_train[0])))
print("\n")
print('y_train: ' + str(scaler.inverse_transform(y_train[0].reshape(-1, 1))) + '\n')

# Build the model
model = keras.Sequential()

model.add(LSTM(units = 700, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.4))

model.add(LSTM(units = 300, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 100))
model.add(Dropout(0.1))

# Output layer
model.add(Dense(units = 1))

# Compiling the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the model to the Training set
history = model.fit(X_train, y_train, epochs = 30, batch_size = 20, validation_split=.30)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Get the stock prices for 2019 to have our model make the predictions
test_data = test_df['Adj Close'].values
test_data = test_data.reshape(-1,1)
test_data = scaler.transform(test_data)

# Create the data to test our model on:
time_steps = 20
X_test, y_test = create_dataset(test_data, time_steps)

# store the original vals for plotting the predictions
y_test = y_test.reshape(-1,1)
org_y = scaler.inverse_transform(y_test)

# reshape it [samples, time steps, features]
X_test = np.reshape(X_test, (X_test.shape[0], time_steps, 1))

# Predict the prices with the model
predicted_y = model.predict(X_test)
predicted_y = scaler.inverse_transform(predicted_y)


# plot the results
plt.plot(org_y, color = 'red', label = 'Real ZS Stock Price')
plt.plot(predicted_y, color = 'blue', label = 'Predicted ZS Stock Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(f'{ticker} Stock Price')
plt.legend()
plt.show()



df_small = _df.loc[:, ['rsi', 'roc_10', 'macd', 'sma20', 'sma50', 'sma200', 'vwap', 'volume', 'adx14', 'bolinger_bands_l20']]
df_small = _df.loc[:, ['rsi', 'roc_10', 'macd','vwap', 'volume', 'adx14']]
n_components= 3
pca = prince.PCA(n_components=n_components, rescale_with_mean=True, rescale_with_std = True, copy=True, check_input = True)
pca.fit(df_small)
fit = pca.fit(df_small)
trans = pca.fit_transform(df_small)
#_df.adjclose.plot()
trans.iloc[:, 0].plot()
plt.show()
print(fit.column_correlations(df_small))




from pyts.decomposition import SingularSpectrumAnalysis
from pymssa import MSSA

window_size = 20
groups = [np.arange(i, i+5) for i in range(0, 20, 5)]

ssa = SingularSpectrumAnalysis(window_size= window_size)
X_ssa = ssa.fit_transform(df_small)

mssa = MSSA(n_components=5,
            window_size=21,
            verbose=True)
mssa.fit(_df.adjclose)

pd.DataFrame(mssa.components_[0,:,:], index=_df.index).plot()
plt.show()

_df.adjclose.plot()
plt.show()
