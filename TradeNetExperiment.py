import os
import time
from copy import deepcopy
from keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from TradeNet import TradeNet
import matplotlib.pyplot as plt

# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")


params = dict()
params['n_steps'] = 14               #Window size or the sequence length
params['lookup_step'] = 3           #Lookup step, 1 is the next day
params['scale'] = True              #whether to scale feature columns & output price as well
params['shuffle'] = True             #whether to shuffle the dataset
params['split_by_time'] = True      #whether to split the training/testing set by date
params['test_size'] = 0.1           #test ratio size, 0.2 is 20%
params['features'] = ["adjclose", "volume", "open", "high", "low"]  #features to use
params['n_layers'] = 6              #Number of RNN layers to use
params['cell'] = LSTM               #Type of RNN to use
params['units'] = 96                #number of cell units
params['dropout'] = 0.2             #node propout percentage
params['loss'] = 'mae'       #Loss function to use - mae or huber_loss
params['optimizer'] = 'adam'        #Optimizer to use
params['batch_size'] = 20           #Size of samples per training session
params['epochs'] = 40               #Number of epochs to run
params['bidirectional'] = True     #whether to use bidirectional RNNs

ticker= 'TQQQ'
bot = TradeNet(ticker, params= params)

# model name to save, making it as unique as possible based on parameters
scale_str = f"sc-{int(bot.SCALE)}"
date_now = time.strftime("%Y-%m-%d")
split_by_date_str = f"sbd-{int(bot.SPLIT_BY_TIME)}"
shuffle_str = f"sh-{int(bot.SHUFFLE)}"

ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-" \
             f"{bot.LOSS}-{bot.OPTIMIZER}-{bot.CELL.__name__}-seq-{bot.N_STEPS}-step-{bot.LOOKUP_STEP}-layers-{bot.N_LAYERS}-units-{bot.UNITS}"

if bot.BIDIRECTIONAL:
    model_name += "-b"

bot.load_data()
bot.create_model()
# train the model and save the weights whenever we see
# a new optimal model using ModelCheckpoint
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

history = bot.model.fit(bot.data["X_train"], bot.data["y_train"],
                        batch_size=bot.BATCH_SIZE,
                        epochs=bot.EPOCHS,
                        validation_data=(bot.data["X_test"], bot.data["y_test"]),
                        callbacks=[checkpointer, tensorboard],
                        verbose=1)

bot.performance_metrics()
bot.plot_graph()

# load optimal model weights from results folder
model_path = os.path.join("results", model_name) + ".h5"
bot.model.load_weights(model_path)
# evaluate the model
loss, mae = bot.model.evaluate(bot.data["X_test"], bot.data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
if bot.SCALE:
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
else:
    mean_absolute_error = mae

# get the final dataframe for the testing set
final_df = get_final_df()
# predict the future price
future_price = predict()