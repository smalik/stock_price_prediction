import os
import numpy as np
import pandas as pd
import subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pandas.plotting import lag_plot, autocorrelation_plot
from pandas import datetime
from statsmodels.tsa import ARIMA
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from pmdarima import auto_arima

class ArimaModel(object):

    def __init__(self, data):
        self.df = data
        self.arima_model = None

    def print_info(self):
        print(self.df.info())
        print(self.df.describe())
        print(self.df.columns)
        print(self.df.head())

    def smape(self, actual, pred):
        abs_diff = np.abs(actual - pred)
        abs_total = np.abs(actual) + np.abs(pred)

        return np.mean((abs_diff*200) / abs_total)

    def find_arima_model(self,
                         data= self.df.adjclose,
                         workers:int = 2,
                         max_p:int = 7,
                         max_q:int= 7,
                         max_P:int = 5,
                         max_Q:int = 5,
                         max_order:int = 20,
                         scoring_method = 'mae'):

        model = auto_arima(y = data,
                           max_p = max_p,
                           max_q= max_q,
                           max_P = max_P,
                           max_Q = max_Q,
                           max_order = max_order,
                           n_jobs= workers,
                           trace= True,
                           return_valid_fits= False,
                           scoring = scoring_method)

        self.arima_model = model


    def train(self, train_set, test_set):

        predictions = list()
        history = [x for x in train_set]

        for t in range(test_set.shape[0]):
            model = ARIMA(history, order=self.model['order'])

