import os
import time
import random
from collections import deque

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import iexfinance as iex
from yahoo_fin import stock_info as si
from matplotlib import pyplot as plt

import stock_data as stok

class TradeNet(object):

    def __init__(self, ticker:str, params:dict):
        self.ticker= ticker
        self.N_STEPS = params['n_steps']
        self.LOOKUP_STEP = params['lookup_step']
        self.SCALE = params['scale']
        self.SHUFFLE = params['shuffle']
        self.SPLIT_BY_TIME = params['split_by_time']
        self.TEST_SIZE = params['test_size']
        self.FEATURE_COLUMNS = params['features']
        self.N_LAYERS = params['n_layers']
        self.CELL = params['cell']
        self.UNITS = params['units']
        self.DROPOUT = params['dropout']
        self.BIDIRECTIONAL = params['bidirectional']
        self.LOSS = params['loss']
        self.OPTIMIZER = params['optimizer']
        self.BATCH_SIZE = params['batch_size']
        self.EPOCHS = params['epochs']
        self.data = None
        self.df = None
        self.model = None

    def shuffle_in_unison(self, a,b):
        # shuffle two arrays in the same way
        state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(state)
        np.random.shuffle(b)

    def load_data(self):
        """
        Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
        Params:
            ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
            n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
            scale (bool): whether to scale prices from 0 to 1, default is True
            shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
            lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
            split_by_date (bool): whether we split the dataset into training/testing by date, setting it
                to False will split datasets in a random way
            test_size (float): ratio for test data, default is 0.2 (20% testing data)
            feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
        """
        # see if ticker is already a loaded stock from yahoo finance
        if isinstance(self.ticker, str):
            # load it from yahoo_fin library
            self.df = si.get_data(self.ticker)
        elif isinstance(self.ticker, pd.DataFrame):
            # already loaded, use it directly
            self.df = self.ticker
        else:
            raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
        # this will contain all the elements we want to return from this function
        result = {}
        # we will also return the original dataframe itself
        result['df'] = self.df.copy()
        # make sure that the passed feature_columns exist in the dataframe
        for col in self.FEATURE_COLUMNS:
            assert col in self.df.columns, f"'{col}' does not exist in the dataframe."
        # add date as a column
        if "date" not in self.df.columns:
            self.df["date"] = self.df.index
        if self.SCALE:
            column_scaler = {}
            # scale the data (prices) from 0 to 1
            for column in self.FEATURE_COLUMNS:
                scaler = preprocessing.MinMaxScaler()
                self.df[column] = scaler.fit_transform(np.expand_dims(self.df[column].values, axis=1))
                column_scaler[column] = scaler
            # add the MinMaxScaler instances to the result returned
            result["column_scaler"] = column_scaler
        # add the target column (label) by shifting by `lookup_step`
        self.df['future'] = self.df['adjclose'].shift(-self.LOOKUP_STEP)
        # last `lookup_step` columns contains NaN in future column
        # get them before droping NaNs
        last_sequence = np.array(self.df[self.FEATURE_COLUMNS].tail(self.LOOKUP_STEP))
        # drop NaNs
        self.df.dropna(inplace=True)
        sequence_data = []
        sequences = deque(maxlen=self.N_STEPS)
        for entry, target in zip(self.df[self.FEATURE_COLUMNS + ["date"]].values, self.df['future'].values):
            sequences.append(entry)
            if len(sequences) == self.N_STEPS:
                sequence_data.append([np.array(sequences), target])
        # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
        # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
        # this last_sequence will be used to predict future stock prices that are not available in the dataset
        last_sequence = list([s[:len(self.FEATURE_COLUMNS)] for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)
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
        if self.SPLIT_BY_TIME:
            # split the dataset into training & testing sets by date (not randomly splitting)
            train_samples = int((1 - self.TEST_SIZE) * len(X))
            result["X_train"] = X[:train_samples]
            result["y_train"] = y[:train_samples]
            result["X_test"] = X[train_samples:]
            result["y_test"] = y[train_samples:]
            if self.SHUFFLE:
                # shuffle the datasets for training (if shuffle parameter is set)
                self.shuffle_in_unison(result["X_train"], result["y_train"])
                self.shuffle_in_unison(result["X_test"], result["y_test"])
        else:
            # split the dataset randomly
            result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                        test_size=self.TEST_SIZE,
                                                                                                        shuffle=self.SHUFFLE)
        # get the list of test set dates
        dates = result["X_test"][:, -1, -1]
        # retrieve test features from the original dataframe
        result["test_df"] = result["df"].loc[dates]
        # remove duplicated dates in the testing dataframe
        result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
        # remove dates from the training/testing sets & convert to float32
        result["X_train"] = result["X_train"][:, :, :len(self.FEATURE_COLUMNS)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(self.FEATURE_COLUMNS)].astype(np.float32)
        self.data = result

        return result

    def create_model(self):

        units = self.UNITS
        n_features = len(self.FEATURE_COLUMNS)
        cell = self.CELL
        sequence_length = self.N_STEPS

        model = Sequential()

        for i in range(self.N_LAYERS):
            if i == 0:
                # first layer
                if self.BIDIRECTIONAL:
                    model.add(Bidirectional(cell(units, return_sequences=True),
                                            batch_input_shape=(None, sequence_length, n_features)))
                else:
                    model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
            elif i == self.N_LAYERS - 1:
                # last layer
                if self.BIDIRECTIONAL:
                    model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                if self.BIDIRECTIONAL:
                    model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    model.add(cell(units, return_sequences=True))
            # add dropout after each layer
            model.add(Dropout(self.DROPOUT))
        model.add(Dense(1, activation="linear"))
        model.compile(loss=self.LOSS, metrics=["mean_absolute_error"], optimizer=self.OPTIMIZER)
        #return model
        self.model = model

    def plot_graph(self):
        """
        This function plots true close price along with predicted close price
        with blue and red colors respectively
        """
        df = self.data['test_df']
        plt.plot(df[f'true_adjclose_{self.LOOKUP_STEP}'], c='b')
        plt.plot(df[f'adjclose_{self.LOOKUP_STEP}'], c='r')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend(["Actual Price", "Predicted Price"])
        plt.show()

    def get_final_df(self):
        """
        This function takes the `model` and `data` dict to
        construct a final dataframe that includes the features along
        with true and predicted prices of the testing dataset
        """
        model = self.model
        data = self.data

        # if predicted future price is higher than the current,
        # then calculate the true future price minus the current price, to get the buy profit
        buy_profit = lambda current, true_future, pred_future: true_future - current if pred_future > current else 0
        # if the predicted future price is lower than the current price,
        # then subtract the true future price from the current price
        sell_profit = lambda current, true_future, pred_future: current - true_future if pred_future < current else 0
        X_test = data["X_test"]
        y_test = data["y_test"]
        # perform prediction and get prices
        y_pred = model.predict(X_test)
        if self.SCALE:
            y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
            y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
        test_df = data["test_df"]
        # add predicted future prices to the dataframe
        test_df[f"adjclose_{self.LOOKUP_STEP}"] = y_pred
        # add true future prices to the dataframe
        test_df[f"true_adjclose_{self.LOOKUP_STEP}"] = y_test
        # sort the dataframe by date
        test_df.sort_index(inplace=True)
        final_df = test_df
        # add the buy profit column
        final_df["buy_profit"] = list(map(buy_profit,
                                          final_df["adjclose"],
                                          final_df[f"adjclose_{self.LOOKUP_STEP}"],
                                          final_df[f"true_adjclose_{self.LOOKUP_STEP}"])
                                      # since we don't have profit for last sequence, add 0's
                                      )
        # add the sell profit column
        final_df["sell_profit"] = list(map(sell_profit,
                                           final_df["adjclose"],
                                           final_df[f"adjclose_{self.LOOKUP_STEP}"],
                                           final_df[f"true_adjclose_{self.LOOKUP_STEP}"])
                                       # since we don't have profit for last sequence, add 0's
                                       )
        self.final_df = final_df
        return final_df

    def predict(self):
        model = self.model
        data = self.data
        # retrieve the last sequence from data
        last_sequence = data["last_sequence"][-self.N_STEPS:]
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        # get the prediction (scaled from 0 to 1)
        prediction = model.predict(last_sequence)
        # get the price (by inverting the scaling)
        if self.SCALE:
            predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
        else:
            predicted_price = prediction[0][0]
        return predicted_price

    def evaluate(self):
        loss, mae = self.model.evaluate(self.data['X_test'], self.data['y_test'])
        if self.SCALE:
            mean_absolute_errer = self.data['column_scaler']['adjclose'].inverse_transform([[mae]])[0][0]
        else:
            mean_absolute_error = mae

        return mae, loss

    def performance_metrics(self):
        final_df = self.get_final_df()
        future_price = self.predict()

        #calculate accuracy by counting the number of positive profits
        num_pos_profit = (final_df[final_df.sell_profit > 0].shape[0]) + (final_df[final_df.buy_profit > 0].shape[0])
        accuracy_score = num_pos_profit/final_df.shape[0]

        total_buy_profit = final_df.buy_profit.sum()
        total_sell_profit = final_df.sell_profit.sum()
        profit_per_trade = (total_sell_profit+total_sell_profit)/final_df.shape[0]

        print(f'Future price after {self.LOOKUP_STEP} days is ${future_price:.2f}')
        print(f'{self.LOSS} loss: {self.evaluate()[1]}')
        print(f'Mean Absolute Error: {self.evaluate()[0]}')
        print(f'Accuracy score: {accuracy_score}')
        print(f'Total buy profit: {total_buy_profit}')
        print(f'Total sell profit: {total_sell_profit}')
        print(f'Total profit: {total_buy_profit+total_sell_profit}')
        print(f'Profit per trade: {profit_per_trade}')

        return accuracy_score, profit_per_trade