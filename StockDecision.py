from datetime import datetime
from TickerData import TickerBase, TSDataGenerator, TSAnalysis, TSModel
from TickerTransform import TickerTransform
import matplotlib.pyplot as plt
import statsmodels as sm
import psycopg2 as pg
from sqlalchemy import create_engine

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

engine = create_engine(('postgresql://plytos:plytos1@192.168.1.7:5432/stockdb'))



symbl = 'BNGO'
ticker = TickerBase(ticker=symbl, start=datetime(2018, 1, 1), engine=engine)
ticker.update_intraday_db_prices()
ticker.fetch_ticker_data()
ticker.fetch_intraday_prices_db()
#ticker.fetch_daily_ticker_db(symbl)
ticker.fetch_ticker_data2()
ticker.compute_indicators()
ticker.compute_candlestick_patterns()

dmodel = TickerTransform(data= ticker.df.adjclose)
dmodel.get_ssa(wsize= 21)
signal = dmodel.retrieve_signal(0.9)
signal.plot()
plt.show()




blah = TSDataGenerator(ticker.df.adjclose, look_back= 60, train_prop= 0.8)
blah = TSDataGenerator(ticker.df, look_back= 60, train_prop= 0.8)
blah.standardize()
train_ds = blah.create_univariate_dataset(data_type= 'train', batch_size=128)
test_ds = blah.create_univariate_dataset(data_type= 'test', batch_size= 128)

predictors = ['open', 'high', 'low', 'adjclose', 'volume', 'beta', 'rsi_14', 'rsi_14_thresh',
              'rsi_21', 'rsi_21_thresh', 'rsi_60', 'rsi_60_thresh', 'macd_12_26',
              'macdsignal_12_26', 'macdhist_12_26', 'macd_5_12', 'macdsignal_5_12',
              'macdhist_5_12', 'mom', 'roc_20', 'roc_125', 'roc_90', 'psar_005',
              'psar_02', 'psar_1', 'psar_005_dist', 'psar_02_dist', 'psar_1_dist',
              'psar_005_ind', 'psar_02_ind', 'psar_1_ind', 'sma20', 'sma50', 'sma200',
              'midprice_5', 'midprice_14', 'midprice_21', 'upperband', 'middleband',
              'lowerband', 'instantaneous_trend', 'adx_7', 'adx_7_pctchg', 'adx_14',
              'adx_14_pctchg', 'adx_21', 'adx_21_pctchg', 'adx_60', 'adx_60_pctchg',
              'cci', 'cci_chg', 'cci_thresh', 'direction_movement_idx', 'money_flow',
              'aroon_down', 'aroon_up', 'ppo', 'trix', 'stoch_k', 'stoch_d',
              'stoch_rsi_k', 'stoch_rsi_d', 'willR', 'natr', 'trange', 'obv', 'adosc',
              'ad', 'log_return', 'percent_return', 'zscore', 'quantile',
              'ht_dom_per', 'ht_dom_cycle', 'ht_trendmode', 'pvt', 'increasing',
              'decreasing', 'cross_sma20', 'cross_sma50', 'cross_sma200',
              'sma20_above', 'sma50_above', 'cross_psar_rsi14',
              'cross_psar_005_close', 'cross_psar_02_close', 'cross_psar_1_close',
              'cross_adx14_psar_02', 'cross_adx7_psar_02', 'cross_stoch',
              'cross_macd_12_26', 'cross_macd_5_12', 'roc_20_125', 'roc_20_125_dist']
blah = TSDataGenerator(ticker.df[predictors].dropna())
blah.standardize()
train_ds = blah.create_multivariate_dataset(data_type= 'train', batch_size= 64)
test_ds = blah.create_multivariate_dataset(data_type= 'test', batch_size= 64)


model = Sequential()
model.add(LSTM(20, return_sequences= True, input_shape=(blah.look_back, 1)))
#model.add(LSTM(20, return_sequences= True, input_shape=(blah.look_back, blah.train['predictors'].shape[1])))
model.add(Dropout(0.1))
model.add(LSTM(20, return_sequences= False))
model.add(Dropout(0.1))
model.add(Dense(5))
model.add(Dense(1))
model.compile(optimizer='adam', loss= 'mape')
history = model.fit_generator(train_ds, epochs=500)

scaler = MinMaxScaler()
scaler.fit(blah.train)
train_len, test_len = blah.lengths['train_size'], blah.lengths['test_size']
model.evaluate_generator(train_ds)
train_predict = model.predict(train_ds, verbose=True)
results = pd.DataFrame(blah.data[blah.look_back:train_len], blah.data.index[blah.look_back:train_len], columns=['actuals'])
results['pred'] = scaler.inverse_transform(train_predict)
results['true'] = blah.data[blah.look_back:train_len]
#results['true'] = blah.data[blah.target][blah.look_back:train_len]
results[['true', 'pred']].plot(subplots= True); plt.show()
results[['true', 'pred']].plot(); plt.show()

scaler = scaler.fit(blah.test)
test_predict = model.predict(test_ds, verbose= True)
start_idx = blah.lengths['test_size']-blah.look_back
projection = pd.DataFrame(blah.data[-(start_idx):], blah.data.index[-(start_idx):], columns=['actuals'])
projection['pred'] = scaler.inverse_transform(test_predict)
projection['true'] = blah.data[-start_idx:]
#projection['true'] = blah.data[blah.target][-start_idx:]
projection[['true', 'pred']].plot(subplots= True); plt.show()
projection[['true', 'pred']].plot(); plt.show()


del projection['actuals']
projection['true_class'] = projection.true.pct_change() > 0
projection['pred_class'] = projection.pred.pct_change() > 0
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(projection.true_class[blah.look_back:], projection.pred_class[blah.look_back:])
cmm_display = ConfusionMatrixDisplay(cm)
cmm_display.plot()
plt.show()


import plotly.graph_objects as go
import plotly.express as px
from plotly import io

io.renderers = ['svg']

fig = go.Figure(
    [go.Scatter(y= projection.true, x= projection.index),
     go.Scatter(y= projection.pred, x= projection.index)]
)
fig.update_xaxes(rangeslider_visible = True,
                rangebreaks=[dict(bounds=["sat", "mon"])]
                )
fig.show()

fig.show()







# RF Directional classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = deepcopy(ticker.df.drop(['close', 'ticker'], axis=1))
#data['target'] = (data.adjclose.pct_change().shift(-1) > data.adjclose.pct_change()).astype(int)
data['target'] = (data.adjclose.pct_change().shift(-1) > 0.06).astype(int)

data = data.dropna()
X_train, X_test, y_train, y_test = train_test_split(data.drop(['target'], axis=1), data.target, test_size=0.1, random_state=42, shuffle=False)
rf = RandomForestClassifier(n_estimators= 1000, bootstrap=False, warm_start=True, n_jobs= 4)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
cm = confusion_matrix(y_test, pred)
cm/np.sum(cm)
print('Correct Prediction (%): ', accuracy_score(y_test, pred, normalize=True)*100.0)
report = classification_report(y_test, pred)
print(report)
returns = deepcopy(X_test)
returns['strategy_returns'] = returns.adjclose.pct_change().shift(-1) * pred
(returns.strategy_returns+1).cumprod().plot();plt.show()