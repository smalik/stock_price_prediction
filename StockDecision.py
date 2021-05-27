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



symbl = 'AAPL'
ticker = TickerBase(ticker=symbl, start=datetime(2018, 1, 1), engine=engine)
#ticker.update_intraday_db_prices()
#ticker.update_daily_db_prices()
ticker.fetch_daily_ticker_db()
ticker.fetch_intraday_prices_db(interval='30t')
ticker.df = ticker.daily.copy()
ticker.compute_indicators()
ticker.compute_candlestick_patterns()


ticker.fetch_ticker_data()
ticker.fetch_ticker_data2()

dmodel = TickerTransform(data= ticker.df.close)
dmodel.get_ssa(wsize= 50)
signal = dmodel.retrieve_signal(0.1)
signal.plot()
plt.show()



lookback= 6
trainprop= 0.6
blah = TSDataGenerator(ticker.df.adjclose, look_back= lookback, train_prop= trainprop)
blah = TSDataGenerator(ticker.df, look_back= lookback, train_prop= trainprop)
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

predictors = ['open', 'high', 'low', 'adjclose', 'volume', 'rsi_14', 'rsi_14_thresh',
              'rsi_21', 'rsi_21_thresh', 'rsi_60', 'rsi_60_thresh', 'macd_12_26',
              'macdsignal_12_26', 'macdhist_12_26', 'macd_5_12', 'macdsignal_5_12',
              'macdhist_5_12', 'mom', 'roc_20', 'roc_125', 'roc_90', 'psar_005',
              'psar_02', 'psar_1', 'psar_005_dist', 'psar_02_dist', 'psar_1_dist',
              'psar_005_ind', 'psar_02_ind', 'psar_1_ind', 'sma20', 'sma50', 'sma200',
              'adx_7', 'adx_7_pctchg', 'adx_14', 'adx_14_pctchg', 'adx_21', 'adx_21_pctchg',
              'adx_60', 'adx_60_pctchg', 'cci', 'cci_chg', 'cci_thresh', 'money_flow',
              'cross_sma20', 'cross_sma50', 'cross_sma200', 'sma20_above', 'sma50_above',
              'cross_stoch', 'cross_macd_12_26', 'cross_macd_5_12', 'roc_20_125', 'roc_20_125_dist']
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
results['pred'] = -scaler.inverse_transform(train_predict)
results['true'] = blah.data[blah.look_back:train_len]
#results['true'] = blah.data[blah.target][blah.look_back:train_len]
results[['true', 'pred']].plot(subplots= True); plt.show()
results[['true', 'pred']].plot(); plt.show()

scaler = scaler.fit(blah.test)
test_predict = model.predict(test_ds, verbose= True)
start_idx = blah.lengths['test_size']-blah.look_back
projection = pd.DataFrame(blah.data[-(start_idx):], blah.data.index[-(start_idx):], columns=['actuals'])
projection['pred'] = -scaler.inverse_transform(test_predict)
projection['true'] = blah.data[-start_idx:]
#projection['true'] = blah.data[-100:]
#projection['true'] = blah.data[blah.target][-start_idx:]
projection[['true', 'pred']].plot(subplots= True); plt.show()
projection[['true', 'pred']].plot(); plt.show()


del projection['actuals']
projection['true_class'] = projection.true.pct_change() > 0
projection['pred_class'] = projection.pred.pct_change() > 0
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
#cm = confusion_matrix(projection.true_class[blah.look_back:], projection.pred_class[blah.look_back:])
cm = confusion_matrix(projection.iloc[-360:,:].true_class[blah.look_back:], projection.iloc[-360:,:].pred_class[blah.look_back:])
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

data = deepcopy(ticker.df[predictors])
#data['target'] = (data.adjclose.pct_change().shift(-1) > data.adjclose.pct_change()).astype(int)
data['target'] = (data.adjclose.pct_change().shift(-1) > 0.06).astype(int)

data = data.dropna()
X_train, X_test, y_train, y_test = train_test_split(data.drop(['target'], axis=1), data.target, test_size=0.1, random_state=42, shuffle=False)
rf = RandomForestClassifier(n_estimators= 1000, bootstrap=False, warm_start=True, n_jobs= 4)
rf.fit(X_train, y_train)
validate = rf.predict(X_train)
pred = rf.predict(X_test)
cm = confusion_matrix(y_test, pred)
cm/np.sum(cm)
print('Correct Prediction (%): ', accuracy_score(y_test, pred, normalize=True)*100.0)
report = classification_report(y_test, pred)
print(report)
returns = deepcopy(X_test)
returns['strategy_returns'] = returns.adjclose.pct_change().shift(-1) * pred
(returns.strategy_returns+1).cumprod().plot();plt.show()


# Singular Spectrum Analysis
rcParams['figure.figsize'] = 24, 24
from mySSA import mySSA
ssa = mySSA(ticker.daily.close)
ssa.embed(embedding_dimension=40, suspected_frequency=suspected_seasonality, verbose=True)
ssa.decompose(verbose=True)
ssa.view_s_contributions(adjust_scale=True)
for i in range(10):
    ssa.view_reconstruction(ssa.Xs[i], names=i, symmetric_plots=i!=0)
plt.show()

ssa.ts.plot(title='Original Time Series'); # This is the original series for comparison
streams5 = [i for i in range(5)]
reconstructed5 = ssa.view_reconstruction(*[ssa.Xs[i] for i in streams5], names=streams5, return_df=True)
ssa.forecast_recurrent(steps_ahead=48, singular_values=streams10, plot=True)

ts_copy5 = ssa.ts.copy()
ts_copy5['Reconstruction'] = reconstructed5.Reconstruction.values
ts_copy5.plot(title='Original vs. Reconstructed Time Series');
plt.show()

# Functioning Discrete Fourer Transform Approach
from pyts.approximation import DiscreteFourierTransform
n_coefs = 21
dft = DiscreteFourierTransform(n_coefs=n_coefs, norm_mean=False, norm_std=False)
X_dft = dft.fit_transform(ticker.daily.adjclose.values.reshape(1,-1))
n_samples, n_timestamps = 1, ticker.daily.shape[0]
if n_coefs % 2 == 0:
    real_idx = np.arange(1, n_coefs, 2)
    imag_idx = np.arange(2, n_coefs, 2)
    X_dft_new = np.c_[
        X_dft[:, :1],
        X_dft[:, real_idx] + 1j * np.c_[X_dft[:, imag_idx],
                                        np.zeros((n_samples, ))]
    ]
else:
    real_idx = np.arange(1, n_coefs, 2)
    imag_idx = np.arange(2, n_coefs + 1, 2)
    X_dft_new = np.c_[
        X_dft[:, :1],
        X_dft[:, real_idx] + 1j * X_dft[:, imag_idx]
    ]
X_irfft = np.fft.irfft(X_dft_new, n_timestamps)
plt.figure(figsize=(24, 12))
plt.plot(ticker.daily.adjclose.values.reshape(1, -1)[0], 'o--', ms=4, label='Original')
plt.plot(X_irfft[0], 'o--', ms=4, label='DFT - {0} coefs'.format(n_coefs))
plt.legend(loc='best', fontsize=10)
plt.xlabel('Time', fontsize=14)
plt.title('Discrete Fourier Transform', fontsize=16)
plt.show()


# Piecewise aggregation approach
from pyts.approximation import PiecewiseAggregateApproximation

X = ticker.daily.adjclose.values.reshape(1,-1)
# PAA transformation
window_size = 10
paa = PiecewiseAggregateApproximation(window_size=window_size)
X_paa = paa.transform(X)

plt.figure(figsize=(24, 12))
plt.plot(X[0], 'o--', ms=4, label='Original')
plt.plot(np.arange(window_size // 2,
                   n_timestamps + window_size // 2,
                   window_size), X_paa[0], 'o--', ms=4, label='PAA')
#plt.vlines(np.arange(0, n_timestamps, window_size) - 0.5, X[0].min(), X[0].max(), color='g', linestyles='--', linewidth=0.5)
plt.legend(loc='best', fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.title('Piecewise Aggregate Approximation', fontsize=16)
plt.show()


# Shaplets Transformation method
from pyts.transformation import ShapeletTransform
predictors = \
    ['rsi_14', 'rsi_14_thresh',
    'rsi_21', 'rsi_21_thresh', 'rsi_60', 'rsi_60_thresh', 'macd_12_26',
    'macdsignal_12_26', 'macdhist_12_26', 'macd_5_12', 'macdsignal_5_12',
    'macdhist_5_12', 'mom', 'roc_20', 'roc_125', 'psar_005',
    'psar_02', 'psar_1', 'psar_005_dist', 'psar_02_dist', 'psar_1_dist',
    'psar_005_ind', 'psar_02_ind', 'psar_1_ind', 'sma20', 'sma50',
    'adx_7', 'adx_14',  'adx_21',
    'adx_60',  'cci', 'cci_chg', 'cci_thresh', 'money_flow',
    'cross_sma20', 'cross_sma50', 'cross_sma200', 'sma20_above', 'sma50_above',
    'cross_stoch', 'cross_macd_12_26', 'cross_macd_5_12', 'roc_20_125', 'adjclose']


X = ticker.indicator_matrix[predictors].dropna(axis=0)
X_train = X[predictors].drop(['adjclose'], axis=1).values
y_train = X['adjclose'].values
st = ShapeletTransform(window_sizes=[3, 5, 10, 14, 21], sort=True)
X_new = st.fit_transform(X_train, y_train)

plt.figure(figsize=(24, 12))
for i, index in enumerate(st.indices_[:4]):
    idx, start, end = index
    plt.plot(X_train[idx], color='C{}'.format(i),
             label='Sample {}'.format(idx))
    plt.plot(np.arange(start, end), X_train[idx, start:end],
             lw=5, color='C{}'.format(i))

plt.xlabel('Time', fontsize=12)
plt.title('The four more discriminative shapelets', fontsize=14)
plt.legend(loc='best', fontsize=8)
plt.show()

# Matrix Profile
import matrixprofile as mp

vals = ticker.daily.adjclose.values
profile, figures = mp.analyze(vals, windows=50)

plt.show()

df = ticker.intraday.copy()
df['strend_signal'] = df.ta.supertrend(3,20).iloc[:,1]
df[['close', 'strend_signal']].plot();plt.show()