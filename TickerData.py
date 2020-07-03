import os
import numpy as np
import time
from datetime import datetime, date
from collections import deque
import random
import ssl
from copy import deepcopy

from iexfinance.stocks import Stock, get_historical_data, get_historical_intraday
from yahoo_fin import stock_info as si

import pandas as pd
import pandas_datareader.data as web
#import pandas_ml as pml

import pandas_ta as ta
import talib
from talib import abstract

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from prince import PCA

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard



if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
ssl._create_default_https_context = ssl._create_unverified_context

class TickerData(object):
    '''
    Object that supports the data representations for stock price modeling
    API pull is either from Yahoo or IEX
    '''
    def __init__(self,
                 ticker:str = None,
                 indicator_list = ['rsi', 'macd', 'volume', 'adjclose', 'vwap',  'bolinger_bands_l20', 'bolinger_bands_m20', 'bolinger_bands_l20']):

        self.df = pd.DataFrame()
        self.ticker = ticker
        self.indicators = indicator_list


        self.all_indicators = \
            rich_features = ['open', 'high', 'low', 'close', 'adjclose', 'volume',
                         'rsi', 'macd', 'adx_14', 'DMP_14', 'DMN_14', 'macdh', 'macds', 'tsi', 'mom',
                         'hl2', 'hlc3', 'ohlc4', 'midpoint', 'midprice',
                         'tema', 'wma10', 'wma20', 'wma30', 'wma50', 'wma200',
                         'sma10', 'sma20', 'sma30', 'sma50', 'sma200', 'ema50', 'ohlc4',
                         'log_return', 'percent_return', 'stdev', 'zscore', 'quantile',
                         'mad', 'adx14', 'eom', 'pvol', 'efi', 'pvt', 'cci', 'increasing',
                         'decreasing', 'bolinger_bands_l20', 'bolinger_bands_m20', 'bolinger_bands_u20',
                         'roc_10', 'stoch_14', 'vwap',
                         'cross_sma10', 'cross_ema50', 'cross_sma200', 'cross_macd']

    def set_api_token(self, token:str= 'pk_dc6d30da3c194003b73023caa63d99a8'):
        self.token = token

    def set_data_endpoint(self, dirpath:str = '/Users/plytos/trading/deep_trading'):
        # # create these folders if they does not exist
        os.chdir(dirpath)
        if not os.path.isdir("results"):
            os.mkdir("results")
        if not os.path.isdir("logs"):
            os.mkdir("logs")
        if not os.path.isdir("data"):
            os.mkdir("data")

    def fetch_ticker_data(self, ticker:str = None) -> pd.DataFrame:
        if isinstance(ticker, str):
            self.df = si.get_data(ticker)
            self.price = self.df.adjclose
        else:
            self.df = si.get_data(self.ticker)
            self.price = self.df.adjclose

    def fetch_ticker_data2(self, ticker:str = None) -> pd.DataFrame:
        start = datetime(2019, 5, 2)
        end = date.today()
        if isinstance(ticker, str):
            self.df = web.DataReader(ticker, 'yahoo', start, end)
        else:
            self.df = web.DataReader(self.ticker, 'yahoo', start, end)

        self.df.columns = ['high', 'low', 'open', 'close', 'volume', 'adjclose']
        self.price = self.df.adjclose

    def get_index_data(self):
        self.spy = si.get_data('SPY')
        self.djia = si.get_data('djia')
        self.idx = si.get_data('IDX')

    def get_sar(high, low, acc: float = 0.02, max: float = 0.1):
        val = talib.SAR(high, low, acceleration=acc, maximum=max)
        return val

    def compute_indicators(self):
        df = self.df
        df = deepcopy(ticker_data.df)
        open = df.open
        close = df.close
        high = df.high
        low = df.low
        volume = df.volume

        nan_offset = 50

        df['beta'] = talib.BETA(high, low, timeperiod=5)
        df['rsi_14'] = talib.RSI(close, timeperiod=14)
        df['rsi_21'] = talib.RSI(close, timeperiod=21)
        df['rsi_60'] = talib.RSI(close, timeperiod=60)
        df['macd_12_26'], df['macdsignal_12_26'], df['macdhist_12_26'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd_5_12'], df['macdsignal_3_15'], df['macdhist_3_15'] = talib.MACD(close, fastperiod=3, slowperiod=15, signalperiod=5)
        df['mom'] = talib.MOM(close, timeperiod=10)
        df['roc_20'] = talib.ROC(close, timeperiod=21)
        df['roc_125'] = talib.ROC(close, timeperiod=125)
        high = deepcopy(df.high)
        low = deepcopy(df.low)
        df['psar_02'] = get_sar(high, low, 0.02, 0.2)
        df['psar_05'] = get_sar(high, low, 0.05, 0.2)
        df['psar_1'] = get_sar(high, low, 0.1, 0.2)
        df['sma20'] = talib.SMA(close, timeperiod=20)
        df['sma50'] = talib.SMA(close, timeperiod=50)
        df['sma200'] = talib.SMA(close, timeperiod=200)
        df['midprice_5'] = talib.MIDPRICE(high, low, timeperiod=5)
        df['midprice_14'] = talib.MIDPRICE(high, low, timeperiod=14)
        df['midprice_21'] = talib.MIDPRICE(high, low, timeperiod=21)
        df['upperband'], df['middleband'], df['lowerband'] = talib.BBANDS(close, timeperiod=5, nbdevup=2., nbdevdn=2., matype=0)
        df['instantaneous_trend'] = talib.HT_TRENDLINE(close)
        df['cci'] = talib.CCI(high, low, close, timeperiod=14)
        df['direction_movement_idx'] = talib.DX(high, low, close, timeperiod=14)
        df['money_flow'] = talib.MFI(high, low, close, volume, timeperiod=14)
        df['ppo'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
        df['trix'] = talib.TRIX(close, timeperiod=30)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['stoch_rsi_k'], df['stoch_rsi_d'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        df['willR'] = talib.WILLR(high, low, close, timeperiod=14)
        df['natr'] = talib.NATR(high, low, close, timeperiod=14)
        df['trange'] = talib.TRANGE(high, low, close)
        df['obv'] = talib.OBV(close, volume)
        df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        df['ad'] = talib.AD(high, low, close, volume)
        df['log_return'] = df.ta.log_return()
        df['percent_return'] = df.ta.percent_return()
        df['zscore'] = df.ta.zscore()
        df['quantile'] = df.ta.quantile()
        df['ht_dom_per'] = talib.HT_DCPERIOD(close)
        df['ht_dom_cycle'] = talib.HT_DCPHASE(close)
        df['ht_trendmode'] = talib.HT_TRENDMODE(close)
        df['pvt'] = df.ta.pvt()
        df['increasing'] = df.ta.increasing()
        df['decreasing'] = df.ta.decreasing()
        #df['cross_sma10'] = df.ta.cross('close', 'sma10')
        #df['cross_sma50'] = df.ta.cross('close', 'sma50')
        #df['cross_sma200'] = df.ta.cross('close', 'sma200')
        #df = df.iloc[nan_offset:, :].dropna(axis=1).copy()

        self.df = df

    def compute_candlestick_patterns(self):
        df = self.df
        open = df.open
        close = df.close
        high = df.high
        low = df.low

        candle_patterns = pd.DataFrame()
        candle_patterns['two_crows'] = CDL2CROWS(open, high, low, close)

    def get_pca(self, components:int = 3):

        data = self.df
        results = dict()
        pca = PCA(n_components = components,
                  n_iter=100,
                  rescale_with_mean = True,
                  rescale_with_std = True,
                  copy = True,
                  check_input = True
                  )
        results['fit'] = pca.fit(data)
        results['rotated'] = pca.fit_transform(data)
        results['feature_correlations'] = fit.column_correlations(data)

        return results




ticker_data = TickerData(ticker='TSLA')
ticker_data.fetch_ticker_data2()
ticker_data.compute_indicators()

class TickerPrediction(object):

    def __init__(self):
        pass

    def create_model(self):
        pass

    def fit(self):
        pass

    def















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
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=test_size,
                                                                                                    shuffle=shuffle)
        # return the result
        return result


