import os
import numpy as np
import time
from datetime import datetime, date, timedelta
from collections import deque
import random
import ssl
from copy import deepcopy

import psycopg2 as pg
import sqlalchemy as sa
from sqlalchemy import create_engine
from iexfinance.stocks import Stock, get_historical_data, get_historical_intraday
from yahoo_fin import stock_info as si

import pandas as pd
import pandas_datareader.data as web
import statsmodels as sm
import psycopg2
# import pandas_ml as pml

import pandas_ta as ta
import talib
from talib import abstract

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from prince import PCA
from pymssa import MSSA

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# from tf.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import keras


class TickerBase(object):
    '''
    Object that supports the data representations for stock price modeling
    API pull is either from Yahoo or IEX
    '''

    def __init__(self,
                 ticker: str = None,
                 start=datetime(2000, 1, 1),
                 end=date.today(),
                 indicator_list=['rsi', 'macd', 'volume', 'adjclose', 'vwap', 'bolinger_bands_l20',
                                 'bolinger_bands_m20', 'bolinger_bands_l20'],
                 engine=None):

        self.df = pd.DataFrame()
        self.daily = pd.DataFrame()
        self.intraday = pd.DataFrame()
        self.target = None
        self.start = start,
        self.end = end,
        self.ticker = ticker
        self.indicators = indicator_list
        self.con = engine
        self.token = None

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

    def set_api_token(self, token: str = 'pk_dc6d30da3c194003b73023caa63d99a8'):
        self.token = token
        os.environ["IEX_API_KEY"] = self.token

    def _get_min_dt(self):
        query = f'select min(timestamp) from {self.ticker.lower()}_price_intraday'
        min_ts = pd.read_sql(query, con=self.con).astype('M8[D]').iloc[0][0]
        print(min_ts.date())
        return min_ts.date()

    def _get_max_dt(self):
        query = f'select max(timestamp) from {self.ticker.lower()}_price_intraday'
        min_ts = pd.read_sql(query, con=self.con).astype('M8[D]').iloc[0][0]
        print(min_ts.date())
        return min_ts.date()

    def _dates_in_table(self, granularity: str = 'daily'):
        query = f'select distinct date(timestamp) from {self.ticker}_price_{granularity}'
        return pd.read_sql(query, con=self.con)

    def _find_missing_dates(self, granularity='daily'):
        prior30_days = [(datetime.today() - timedelta(i)).date() for i in range(0,30)]
        days_in_db = [item.item() for item in self._dates_in_table(granularity=granularity).values]
        days_to_fetch = sorted(set(prior30_days) - set(days_in_db))

        return days_to_fetch


    def set_data_endpoint(self, dirpath: str = '/Users/plytos/trading/deep_trading'):
        # # create these folders if they does not exist
        os.chdir(dirpath)
        if not os.path.isdir("results"):
            os.mkdir("results")
        if not os.path.isdir("logs"):
            os.mkdir("logs")
        if not os.path.isdir("data"):
            os.mkdir("data")

    # getCompanyInfo returns a dictionary with the company symbol as a key and the info as the value
    # call to iex finance api to return company info for a list of symbols
    def getCompanyInfo(self, symbols):
        stock_batch = Stock(symbols,
                            token=self.token)
        company_info = stock_batch.get_company()
        return company_info

    # getEarnings returns quarterly earnings
    def getEarnings(self, symbol):
        stock_batch = Stock(symbol,
                            token=self.token)
        earnings = stock_batch.get_earnings(last=12)
        return earnings

    def getHistoricalPrices(self, start= None, end= None):
        if start is None:
            start = self.start
        else:
            start = start
        if end is None:
            end = self.end
        else:
            end = end
        prices = get_historical_data(self.ticker,
                                     start,
                                     end,
                                     close_only=False,
                                     output_format='pandas',
                                     token=self.token)
        self.df = prices

    def update_daily_db_prices(self, update_type:str = 'append', ticker_override= None):
        if ticker_override is None:
            ticker = self.ticker
        else:
            ticker = ticker_override
        print(f'Ticker now set to {ticker}')


        if self.token is None:
            self.set_api_token()

        try:
            query = f'select max(timestamp) from {self.ticker.lower()}_price_daily'
            last_date = pd.read_sql(query, con=self.con)
            rowcount = pd.read_sql(f'select count(*) from {self.ticker.lower()}_price_daily', con=self.con)
            if np.max(rowcount).values == 0:
                month = pd.DatetimeIndex(last_date.values[0]).month.astype(int)[0]
                day = pd.DatetimeIndex(last_date.values[0]).day.astype(int)[0] + 1
                year = pd.DatetimeIndex(last_date.values[0]).year.astype(int)[0]
                ref_time = datetime(year, month, day)
                prices = get_historical_data(ticker,
                                             ref_time,
                                             self.end[0],
                                             close_only=False,
                                             output_format='pandas',
                                             token=self.token)
            else:
                #if ref_time.date() >= datetime.datetime.today().date():
                    #ref_time = datetime.now() - timedelta(days=15 * 365)
                update_type = 'replace'
                ref_time = datetime.now() - timedelta(days=15 * 365)
                prices = get_historical_data(ticker,
                                             ref_time,
                                             self.end[0],
                                             close_only=False,
                                             output_format='pandas',
                                             token=self.token)
        except Exception as e:
            print(e)
            update_type = 'replace'
            ref_time = datetime.now() - timedelta(days=15*365)
            prices = get_historical_data(ticker,
                                         ref_time,
                                         self.end[0],
                                         close_only=False,
                                         output_format='pandas',
                                         token=self.token)
        print(f'Writing to table {ticker.lower()}_price_daily')

        prices.to_sql(f'{ticker.lower()}_price_daily', con=self.con, if_exists=update_type, index_label='timestamp')

    def update_all_daily_db_prices(self):
        query_tblnames = ''' 
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
        '''

        tbls = pd.read_sql(query_tblnames, self.con)

        tickers = [ticker.split('_')[0].upper() for ticker in tbls.table_name]
        print(tickers)

        for ticker in tickers:
            try:
                print(f'Updating daily price table for : {ticker}...')
                self.update_daily_db_prices(ticker_override= ticker)
                print(f'{ticker} prices updated.')
            except Exception as e:
                print(e)

    def getIntradayPrices(self, date, stock='QQQ'):
        prices = get_historical_intraday(stock,
                                         date,
                                         output_format='pandas',
                                         token=self.token)
        return prices

    def update_intraday_db_prices(self, update_type: str= 'append'):
        if self.token is None:
            self.set_api_token()
        print(self.token)
        try:
            fetch_dates = self._find_missing_dates(granularity='intraday')
            prices = self.getIntradayPrices(date=fetch_dates[0], stock= self.ticker)
            for day in fetch_dates[1:]:
                print(f'Fetching intraday price data for {day}')
                prices = prices.append(self.getIntradayPrices(date= day, stock= self.ticker))
            print(prices.shape)
            prices.to_sql(f'{self.ticker.lower()}_price_intraday', con=self.con, if_exists=update_type, index_label='timestamp')
        except:
            print('UndefinedTable exception triggered')
            prior30_days = [(datetime.today() - timedelta(i)).date() for i in range(0,30)]
            prices = self.getIntradayPrices(date=prior30_days[0], stock=self.ticker)
            for day in prior30_days[1:]:
                print(f'Fetching intraday price data for {day}')
                prices = prices.append(self.getIntradayPrices(date=day, stock=self.ticker))
            print(prices.shape)
            prices.to_sql(f'{self.ticker.lower()}_price_intraday', con=self.con, if_exists=update_type,
                          index_label='timestamp')

    def get_agg_prices(self, measure: str='close' , interval: str='30m', agg_type = np.mean):
        query = f'select * from {self.ticker.lower()}_price_intraday'
        data = pd.read_sql(query, con=self.con).sort_index()
        return data[measure].resample(interval).apply(agg_type)

    def fetch_intraday_prices_db(self, value: str = 'all', interval: str = '30T', aggfunc= np.mean):
        query = f'select * from {self.ticker.lower()}_price_intraday'
        data = pd.read_sql(query, con= self.con, index_col='timestamp')
        data.rename(columns={'close': 'adjclose'}, inplace= True)
        pdata = data.sort_index()
        if value != 'all':
            pdata = pdata[value].resample(interval).apply(aggfunc).dropna()

        else:
            pdata = pdata.resample(interval).apply(aggfunc).dropna()

        pdata['close'] = pdata.adjclose.copy()

        self.df = pdata
        self.intraday = pdata

        return pdata

    def fetch_daily_ticker_db(self, ticker=None):
        if ticker is None:
            ticker = self.ticker
        else:
            ticker = ticker

        engine = create_engine(('postgresql://plytos:plytos1@192.168.1.7:5432/stockdb'))

        query= f'select * from {ticker.lower()}_price_daily'
        data = pd.read_sql(query, con=engine, index_col='timestamp')

        convert_dict = dict()
        convert_dict['close'] = float
        convert_dict['high'] = float
        convert_dict['low'] = float
        convert_dict['open'] = float
        convert_dict['volume'] = float
        convert_dict['changeOverTime'] = float
        convert_dict['marketChangeOverTime'] = float
        convert_dict['uClose'] = float
        convert_dict['uOpen'] = float
        convert_dict['uHigh'] = float
        convert_dict['uLow'] = float
        convert_dict['uVolume'] = float
        convert_dict['fClose'] = float
        convert_dict['fOpen'] = float
        convert_dict['fHigh'] = float
        convert_dict['fLow'] = float
        convert_dict['fVolume'] = float
        convert_dict['change'] = float
        convert_dict['changePercent'] = float
        data = data.astype(convert_dict)
        data['adjclose'] = data.close.copy()

        self.data = data
        self.daily = data


    def fetch_ticker_data(self, ticker: str = None) -> pd.DataFrame:
        if isinstance(ticker, str):
            self.df = si.get_data(ticker)
            self.price = self.df.adjclose
        else:
            self.df = si.get_data(self.ticker)
            self.price = self.df.adjclose

    def fetch_ticker_data2(self, ticker: str = None) -> pd.DataFrame:
        start = self.start[0]
        end = self.end[0]
        if isinstance(ticker, str):
            self.df = web.DataReader(ticker, 'yahoo', start, end)
        else:
            self.df = web.DataReader(self.ticker, 'yahoo', start, end)

        self.df.columns = ['high', 'low', 'open', 'close', 'volume', 'adjclose']
        self.price = self.df.adjclose

    def load_data_from_csv(self, fpath):
        self.df = pd.read_csv(filepath_or_buffer=fpath)

    def get_index_data(self):
        self.spy = si.get_data('SPY')
        self.djia = si.get_data('djia')
        self.idx = si.get_data('IDX')

    def standardize(self, col='adjclose', type: str = 'standard'):
        if type == 'standard':
            scaler = StandardScaler()
        elif type == 'min-max':
            scaler = MinMaxScaler(feature_range=(0, 1))
        elif type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        self.scaled_data = pd.DataFrame(scaler.fit_transform(self.df[col].values.reshape(-1, 1)),
                                        index=self.df.index,
                                        columns=[f'{col}_norm'])

    def get_sar(self, high, low, acc: float = 0.02, max: float = 0.1):
        val = talib.SAR(high, low, acceleration=acc, maximum=max)
        return val

    def confluence_tally(self):
        pass

    def _calculate_ichimoku(self):
        data = self.df[['open', 'high', 'low', 'close']]

        # Tenkan-sen (conversion line): ((9 period high + 9 period low)/2)
        nine_period_high = data.high.rolling(window=9).max()
        nine_period_low = data.low.rolling(window= 9).max()
        data['tenkan_sen'] = (nine_period_high + nine_period_low)/2

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
        period26_high = data.high.rolling(window=26).max()
        period26_low = data.low.rolling(window=26).min()
        data['kijun_sen'] = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
        period52_high = data.high.rolling(window=52).max()
        period52_low = data.low.rolling(window=52).min()
        data['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(52)

        # The most current closing price plotted 26 time periods behind (optional)
        data['chikou_span'] = data['close'].shift(-26)

        data.dropna(inplace=True)

        return data

    def _ichimoku_strategy(self, cloudframe:pd.DataFrame):
        strategy = pd.DataFrame(0, index = cloudframe.index, columns= ['above_cloud', 'A_above_B', 'tenkan_kiju_cross'])

    def compute_indicators(self):
        df = deepcopy(self.df)
        open = df.open
        close = df.close
        high = df.high
        low = df.low
        volume = df.volume
        adjclose = close

        nan_offset = 50

        df['beta'] = talib.BETA(high, low, timeperiod=5)
        df['rsi_14'] = talib.RSI(close, timeperiod=14)
        df['rsi_14_thresh'] = df.rsi_14.apply(lambda c: 1 if c > 70 else (-1 if c < 30 else 0))
        df['rsi_21'] = talib.RSI(close, timeperiod=21)
        df['rsi_21_thresh'] = df.rsi_21.apply(lambda c: 1 if c > 70 else (-1 if c < 30 else 0))
        df['rsi_60'] = talib.RSI(close, timeperiod=60)
        df['rsi_60_thresh'] = df.rsi_60.apply(lambda c: 1 if c > 70 else (-1 if c < 30 else 0))
        df['macd_12_26'], df['macdsignal_12_26'], df['macdhist_12_26'] = talib.MACD(close, fastperiod=12, slowperiod=26,
                                                                                    signalperiod=9)
        df['macd_5_12'], df['macdsignal_5_12'], df['macdhist_5_12'] = talib.MACD(close, fastperiod=3, slowperiod=15,
                                                                                 signalperiod=5)
        df['mom'] = talib.MOM(close, timeperiod=10)
        df['roc_20'] = talib.ROC(close, timeperiod=21)
        df['roc_125'] = talib.ROC(close, timeperiod=125)
        df['roc_90'] = talib.ROC(close, timeperiod=90)
        high = deepcopy(df.high)
        low = deepcopy(df.low)
        df['psar_005'] = self.get_sar(high, low, 0.005, 0.2)
        df['psar_02'] = self.get_sar(high, low, 0.02, 0.2)
        df['psar_1'] = self.get_sar(high, low, 0.1, 0.2)
        try:
            df['psar_005_dist'] = df.psar_005 - df.adjclose
            df['psar_02_dist'] = df.psar_02 - df.adjclose
            df['psar_1_dist'] = df.psar_1 - df.adjclose
        except Exception as e:
            print(e)
        try:
            df['psar_005_ind'] = (df.psar_005 < self.df.adjclose).astype(int)
            df['psar_02_ind'] = (df.psar_02 < self.df.adjclose).astype(int)
            df['psar_1_ind'] = (df.psar_1 < self.df.adjclose).astype(int)
        except Exception as e:
            print(e)
        df['sma5'] = talib.SMA(close, timeperiod=5)
        df['sma10'] = talib.SMA(close, timeperiod=10)
        df['sma20'] = talib.SMA(close, timeperiod=20)
        df['sma50'] = talib.SMA(close, timeperiod=50)
        df['sma200'] = talib.SMA(close, timeperiod=200)
        df['midprice_5'] = talib.MIDPRICE(high, low, timeperiod=5)
        df['midprice_14'] = talib.MIDPRICE(high, low, timeperiod=14)
        df['midprice_21'] = talib.MIDPRICE(high, low, timeperiod=21)
        df['upperband'], df['middleband'], df['lowerband'] = talib.BBANDS(close, timeperiod=5, nbdevup=2., nbdevdn=2.,
                                                                          matype=0)
        df['instantaneous_trend'] = talib.HT_TRENDLINE(close)
        df['adx_7'] = talib.ADX(high, low, close, timeperiod=7)
        df['adx_7_pctchg'] = df.adx_7.pct_change()
        df['adx_14'] = talib.ADX(high, low, close, timeperiod=14)
        df['adx_14_pctchg'] = df.adx_14.pct_change()
        df['adx_21'] = talib.ADX(high, low, close, timeperiod=21)
        df['adx_21_pctchg'] = df.adx_21.pct_change()
        df['adx_60'] = talib.ADX(high, low, close, timeperiod=61)
        df['adx_60_pctchg'] = df.adx_60.pct_change()
        df['cci'] = talib.CCI(high, low, close, timeperiod=14)
        df['cci_chg'] = df.cci.pct_change()
        df['cci_thresh'] = df.cci.apply(lambda c: 1 if c > 100 else (-1 if c < 20 else 0))
        df['direction_movement_idx'] = talib.DX(high, low, close, timeperiod=14)
        df['money_flow'] = talib.MFI(high, low, close, volume, timeperiod=14)
        df['aroon_down'], df['aroon_up'] = talib.AROON(high, low)
        df['ppo'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
        df['trix'] = talib.TRIX(close, timeperiod=30)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0,
                                                   slowd_period=3, slowd_matype=0)
        df['stoch_rsi_k'], df['stoch_rsi_d'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3,
                                                              fastd_matype=0)
        df['willR'] = talib.WILLR(high, low, close, timeperiod=14)
        df['natr'] = talib.NATR(high, low, close, timeperiod=14)
        df['trange'] = talib.TRANGE(high, low, close)
        df['obv'] = talib.OBV(close, volume)
        df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        df['ad'] = talib.AD(high, low, close, volume)
        try:
            df['log_return'] = df.ta.log_return()
            df['percent_return'] = df.ta.percent_return()
        except Exception as e:
            print(e)
        df['zscore'] = df.ta.zscore()
        df['quantile'] = df.ta.quantile()
        df['ht_dom_per'] = talib.HT_DCPERIOD(close)
        df['ht_dom_cycle'] = talib.HT_DCPHASE(close)
        df['ht_trendmode'] = talib.HT_TRENDMODE(close)
        df['pvt'] = df.ta.pvt()
        df['increasing'] = df.ta.increasing()
        df['decreasing'] = df.ta.decreasing()
        df['cross_sma5'] = df.ta.cross('close', 'sma5')
        df['cross_sma10'] = df.ta.cross('close', 'sma10')
        df['cross_sma20'] = df.ta.cross('close', 'sma20')
        df['cross_sma50'] = df.ta.cross('close', 'sma50')
        df['cross_sma200'] = df.ta.cross('close', 'sma200')
        df['sma20_above'] = (df.sma20 > df.sma200).astype(int)
        df['sma50_above'] = (df.sma50 > df.sma200).astype(int)
        df['cross_psar_rsi14'] = df.ta.cross('psar_02', 'rsi_14')
        df['cross_psar_005_close'] = df.ta.cross('psar_005', 'adjclose')
        df['cross_psar_02_close'] = df.ta.cross('psar_02', 'adjclose')
        df['cross_psar_1_close'] = df.ta.cross('psar_1', 'adjclose')
        df['cross_adx14_psar_02'] = df.ta.cross('psar_02', 'adx_14')
        df['cross_adx7_psar_02'] = df.ta.cross('psar_005', 'adx_7')
        df['cross_adx14_psar_02'] = df.ta.cross('psar_005', 'adx_14')
        df['cross_stoch'] = df.ta.cross('stoch_k', 'stoch_d')
        df['cross_macd_12_26'] = df.ta.cross('macd_12_26', 'macdsignal_12_26')
        df['cross_macd_5_12'] = df.ta.cross('macd_5_12', 'macdsignal_5_12')
        df['roc_20_125'] = df.ta.cross('roc_20', 'roc_125')
        df['roc_20_125_dist'] = df.roc_20 - df.roc_125

        #df = df.iloc[nan_offset:, :].dropna(axis=1).copy()

        self.df = df
        self.indicator_matrix = df.iloc[:, 6:]

    def compute_candlestick_patterns(self):
        df = self.df
        open = df.open
        close = df.close
        high = df.high
        low = df.low

        candle_patterns = pd.DataFrame(deepcopy(close))
        candle_patterns['two_crows'] = talib.CDL2CROWS(open, high, low, close)
        candle_patterns['three_black_crows'] = talib.CDL3BLACKCROWS(open, high, low, close)
        candle_patterns['three_inside'] = talib.CDL3INSIDE(open, high, low, close)
        candle_patterns['three_line_strike'] = talib.CDL3LINESTRIKE(open, high, low, close)
        candle_patterns['three_outside'] = talib.CDL3OUTSIDE(open, high, low, close)
        candle_patterns['three_star_south'] = talib.CDL3STARSINSOUTH(open, high, low, close)
        candle_patterns['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open, high, low, close)
        candle_patterns['abandoned_baby'] = talib.CDLABANDONEDBABY(open, high, low, close)
        candle_patterns['advance_block'] = talib.CDLADVANCEBLOCK(open, high, low, close)
        candle_patterns['belt_hold'] = talib.CDLBELTHOLD(open, high, low, close)
        candle_patterns['breakaway'] = talib.CDLBREAKAWAY(open, high, low, close)
        candle_patterns['closing_marubozu'] = talib.CDLCLOSINGMARUBOZU(open, high, low, close)
        candle_patterns['concealing_baby_swallow'] = talib.CDLCONCEALBABYSWALL(open, high, low, close)
        candle_patterns['counterattack'] = talib.CDLCOUNTERATTACK(open, high, low, close)
        candle_patterns['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(open, high, low, close)
        candle_patterns['doji'] = talib.CDLDOJI(open, high, low, close)
        candle_patterns['doji_star'] = talib.CDLDOJISTAR(open, high, low, close)
        candle_patterns['gravestone_doji'] = talib.CDLGRAVESTONEDOJI(open, high, low, close)
        candle_patterns['dragonfly_doji'] = talib.CDLDRAGONFLYDOJI(open, high, low, close)
        candle_patterns['engulfing'] = talib.CDLENGULFING(open, high, low, close)
        candle_patterns['eveningstar_doji'] = talib.CDLEVENINGDOJISTAR(open, high, low, close)
        candle_patterns['eveningstar'] = talib.CDLEVENINGSTAR(open, high, low, close)
        candle_patterns['gap_side_white'] = talib.CDLGAPSIDESIDEWHITE(open, high, low, close)
        candle_patterns['hammer'] = talib.CDLHAMMER(open, high, low, close)
        candle_patterns['doji'] = talib.CDLDOJI(open, high, low, close)
        candle_patterns['hanging_man'] = talib.CDLHANGINGMAN(open, high, low, close)
        candle_patterns['harami'] = talib.CDLHARAMI(open, high, low, close)
        candle_patterns['harami_cross'] = talib.CDLHARAMICROSS(open, high, low, close)
        candle_patterns['high_wave'] = talib.CDLHIGHWAVE(open, high, low, close)
        candle_patterns['hikkake'] = talib.CDLHIKKAKE(open, high, low, close)
        candle_patterns['hikkake_mod'] = talib.CDLHIKKAKEMOD(open, high, low, close)
        candle_patterns['homing_pigeon'] = talib.CDLHOMINGPIGEON(open, high, low, close)
        candle_patterns['identical_3crows'] = talib.CDLIDENTICAL3CROWS(open, high, low, close)
        candle_patterns['in_neck'] = talib.CDLINNECK(open, high, low, close)
        candle_patterns['inverted_hammer'] = talib.CDLINVERTEDHAMMER(open, high, low, close)
        candle_patterns['kicking'] = talib.CDLKICKING(open, high, low, close)
        candle_patterns['kicking_marubozu'] = talib.CDLKICKINGBYLENGTH(open, high, low, close)
        candle_patterns['ladder_bottom'] = talib.CDLLADDERBOTTOM(open, high, low, close)
        candle_patterns['long_leg_doji'] = talib.CDLLONGLEGGEDDOJI(open, high, low, close)
        candle_patterns['long_line'] = talib.CDLLONGLINE(open, high, low, close)
        candle_patterns['marubozu'] = talib.CDLMARUBOZU(open, high, low, close)
        candle_patterns['matching_low'] = talib.CDLMATCHINGLOW(open, high, low, close)
        candle_patterns['mat_hold'] = talib.CDLMATHOLD(open, high, low, close)
        candle_patterns['morningstar_doji'] = talib.CDLMORNINGDOJISTAR(open, high, low, close)
        candle_patterns['morningstar'] = talib.CDLMORNINGSTAR(open, high, low, close)
        candle_patterns['on_neck'] = talib.CDLONNECK(open, high, low, close)
        candle_patterns['piercing'] = talib.CDLPIERCING(open, high, low, close)
        candle_patterns['rickshaw_man'] = talib.CDLRICKSHAWMAN(open, high, low, close)
        candle_patterns['rising_fall_3methods'] = talib.CDLRISEFALL3METHODS(open, high, low, close)
        candle_patterns['separating_lines'] = talib.CDLSEPARATINGLINES(open, high, low, close)
        candle_patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(open, high, low, close)
        candle_patterns['short_line'] = talib.CDLSHORTLINE(open, high, low, close)
        candle_patterns['spinning_top'] = talib.CDLSPINNINGTOP(open, high, low, close)
        candle_patterns['stalled'] = talib.CDLSTALLEDPATTERN(open, high, low, close)
        candle_patterns['stick_sandwich'] = talib.CDLSTICKSANDWICH(open, high, low, close)
        candle_patterns['takuri'] = talib.CDLTAKURI(open, high, low, close)
        candle_patterns['tasuki_gap'] = talib.CDLTASUKIGAP(open, high, low, close)
        candle_patterns['thrusting'] = talib.CDLTHRUSTING(open, high, low, close)
        candle_patterns['tristar'] = talib.CDLTRISTAR(open, high, low, close)
        candle_patterns['unique_3river'] = talib.CDLUNIQUE3RIVER(open, high, low, close)
        candle_patterns['upside_gap_2crows'] = talib.CDLUPSIDEGAP2CROWS(open, high, low, close)
        candle_patterns['upside_gap_3methods'] = talib.CDLXSIDEGAP3METHODS(open, high, low, close)

        self.candlestick_patterns = candle_patterns

    def _get_slope(self, indicator:str= 'adjclose'):

        return pd.Series(np.gradient(self.df[indicator].values), self.df.index, name=f'slope_{indicator}')

    def append_change_column(self, df, ticker):
        """
        Append new pct_chg column and new close column to the main dataset for ticker of interest
        :param df:
        :param ticker:
        :return:
        """
        df2 = pd.DataFrame()
        df2['change'] = np.log(df['close']) - np.log(df['close'].shift(1))
        self.df[str(ticker) + 'CHG'] = df2['change']
        self.df[str(ticker) + 'CLS'] = df['close']

        return self.df

    def backTester(self, df):
        for x in range(len(df.columns) - 2):
            df['stock' + str(x + 1)]
            df['stock1compair'] = np.where(df['stock' + str(x + 1)].values < df['stock' + str(x)].values and
                                           df['stock' + str(x + 1)].values < df['stock' + str(x + 2)].values, 1, 0)
        return df

    def create_learning_dataset_multi(self,
                                      train_prop: float = 0.9,
                                      time_step: int = 1):
        data = self.df

    def create_dataset(self, look_back: int = 60):
        '''
        Function to create a dataset to feed into an LSTM
        '''

        dataset = self.df.values
        scaler = MinMaxScaler()
        scaler.fit(dataset)
        dataset = scaler.transform(dataset)

        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def make_train_test(self, time_step: int = 60):
        x_train, y_train = self.create_dataset(time_step)
        return x_train, y_train

    def ts_to_supervised(self, data: pd.DataFrame, n_lag: int = 60, n_out=1, dropna: bool = True):
        """
        Transform a univariate series to a supervised learning dataset
        :param data: time series values
        :return: pandas dataframe
        """
        n_vars = data.shape[1]
        df = deepcopy(data)
        cols, names = list(), list()

        # for i in range(n_in, 0, -1):

    # TODO: Write code for Ichimoky cloud
    # TODO: Add Ichimoku indicators and rules to dataset
    # TODO: Write ML method to discover support and resistance levels for chart
    # TODO: Methods to generate train, test, and validation datasets
    # TODO: Method to create dataset for neural network


class TSDataGenerator(object):

    def __init__(self,
                 df,
                 target: str = 'adjclose',
                 train_prop: float = 0.67,
                 look_back: int = 20):
        self.data = df
        self.target = target
        self.scaled_data = None
        self.look_back = look_back
        self.train_prop = train_prop
        self.train = dict()
        self.test = dict()
        self._train = None
        self._test = None
        self.lengths = dict()
        self.start_idx = None

    def _set_index(self):
        self.start_idx = self.lengths['test_size'] - self.look_back

    def standardize(self):

        data = self.data
        scaler = MinMaxScaler(feature_range=(0, 1))

        if isinstance(data, pd.Series):
            self.lengths['train_size'] = int(len(data) * self.train_prop)
            self.lengths['test_size'] = len(data) - self.lengths['train_size']

            self.scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
            print(f'In standardize() Series method.\n')
            print(f'test side: {self.lengths["test_size"]}\n')
            print(f'train_size: {self.lengths["train_size"]}')

            self.train = self.scaled_data[:self.lengths['train_size']]
            self.test = self.scaled_data[-self.lengths['test_size']:self.data.shape[0]]

        elif isinstance(data, pd.DataFrame):
            outcome_idx = self.data.columns.get_loc(self.target)
            self.lengths['train_size'] = int(self.data.shape[0] * self.train_prop)
            self.lengths['test_size'] = self.data.shape[0] - self.lengths['train_size']
            self.scaled_data = scaler.fit_transform(self.data.values)
            print(f'In standardize() DataFrame method.\n '
                  f'test size: {self.lengths["test_size"]}\n '
                  f'train_size: {self.lengths["train_size"]}')

            train = dict()
            tmp = self.scaled_data[: self.lengths['train_size'], :]
            train['predictors'] = np.delete(tmp, [outcome_idx], axis=1)
            # data.iloc[:train_size, :].drop([self.target], axis=1)
            train['outcome'] = self.scaled_data[: self.lengths['train_size'], outcome_idx]
            # data.iloc[:train_size, outcome_idx]

            test = dict()
            tmp = self.scaled_data[-self.lengths['test_size']:, ]
            test['predictors'] = np.delete(tmp, [outcome_idx], axis=1)
            # data.iloc[test_size:len(data), :].drop([self.target], axis=1)
            test['outcome'] = self.scaled_data[-self.lengths['test_size']:, outcome_idx]
            # data.iloc[test_size:len(data), outcome.idx]

            self.train = train
            self.test = test

            '''
            self.train, self.test = data
            self.train['predictors'] = self.train[0]
            self.test['predictors'] = selftest[0]
            self.train['outcome'] = self.train[1]
            self.test['outcome'] = self.test[0]
            '''

    def create_univariate_dataset(self,
                                  data_type: str = 'train',
                                  sampling_rate: int = 1,
                                  stride: int = 1,
                                  batch_size: int = 1):

        if data_type == 'train':
            data = self.train
        elif data_type == 'test':
            data = self.test

        generator = TimeseriesGenerator(data,
                                        data,
                                        length=self.look_back,
                                        sampling_rate=sampling_rate,
                                        stride=stride,
                                        batch_size=batch_size)
        return generator

    def create_multivariate_dataset(self,
                                    data_type='train',
                                    outcome: str = 'adjclose',
                                    sampling_rate: int = 1,
                                    stride: int = 1,
                                    batch_size: int = 1):

        if data_type == 'train':
            outcome = self.train['outcome']
            data = self.train['predictors']
        elif data_type == 'test':
            outcome = self.test['outcome']
            data = self.test['predictors']

        generator = TimeseriesGenerator(data,
                                        outcome,
                                        length=self.look_back,
                                        sampling_rate=sampling_rate,
                                        stride=stride,
                                        batch_size=batch_size)
        return generator

    def run_model_test(self, epochs: int = 10):
        model = Sequential()
        model.add(LSTM(10, input_shape=(self.look_back, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        history = model.fit_generator(self.train, epochs=epochs)

        model.evaluate_generator(test_data_gen)
        trainPredict = model.predict_generator(train_data_gen)
        testPredict = model.predict_generator(test_data_gen)

        return history

class TSAnalysis(object):
    def __init__(self, data):
        self.data = data
        self.model = None

    def _find_regime_change(self, regimes: int = 3, switching_variance: bool = True):
        model = sm.tsa.MarkovRegression(self.data,
                                        k_regimes=regimes,
                                        trend='nc',
                                        switching_variance=switching_variance)
        fitted = model.fit()
        return fitted

    def plot_regime_change(self, model):
        fig, axes = plt.subplots(3, figsize=(10, 7))
        ax = axes[0]
        ax.plot(model.smoothed_marginal_probabilities[0])
        ax.set(title='Smoothed probability of a low-variance regime for stock returns')
        ax = axes[1]
        ax.plot(model.smoothed_marginal_probabilities[1])
        ax.set(title='Smoothed probability of a medium-variance regime for stock returns')
        ax = axes[2]
        ax.plot(model.smoothed_marginal_probabilities[2])
        ax.set(title='Smoothed probability of a high-variance regime for stock returns')
        fig.tight_layout()
        plt.show()


class TSModel(object):

    def __init__(self, data):
        self.data = data
        self.model = None


    # Create deep learning tensorflow model
    def create_sequential_model(sequence_length,
                     units=256,
                     cell=LSTM,
                     n_layers=2,
                     dropout=0.3,
                     loss="mean_absolute_error",
                     optimizer="rmsprop",
                     bidirectional=False):

        model = Sequential()
        for i in range(n_layers):
            if i == 0:
                # first layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(None, sequence_length)))
                else:
                    model.add(cell(units, return_sequences=True, input_shape=(None, sequence_length)))
            elif i == n_layers - 1:
                # last layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    model.add(cell(units, return_sequences=True))
            # add dropout after each layer
            model.add(Dropout(dropout))
        model.add(Dense(1, activation="linear"))
        model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
        return model

    # predict future price using trained deep learning model
    def predict(model, data, classification=False):
        # retrieve the last sequence from data
        last_sequence = data["last_sequence"][:N_STEPS]
        # retrieve the column scalers
        column_scaler = data["column_scaler"]
        # reshape the last sequence
        last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        # get the prediction (scaled from 0 to 1)
        prediction = model.predict(last_sequence)
        # get the price (by inverting the scaling)
        predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
        return predicted_price

    # get accuracy for model
    def get_accuracy(model, data):
        y_test = data["y_test"]
        X_test = data["X_test"]
        y_pred = model.predict(X_test)
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
        y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
        y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
        return accuracy_score(y_test, y_pred)

    # Plot model prediction vs actuals
    def plot_graph(model, data):
        y_test = data["y_test"]
        X_test = data["X_test"]
        y_pred = model.predict(X_test)
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
        # last 200 days, feel free to edit that
        plt.plot(y_test[-200:], c='b')
        plt.plot(y_pred[-200:], c='r')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend(["Actual Price", "Predicted Price"])
        plt.show()