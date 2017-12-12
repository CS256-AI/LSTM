import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import h5py
import matplotlib.pyplot as plt


def global_normalize(x):
    x_base = x.max(axis=0)
    x = x/x_base
    return x, x_base


def percentage_normalize(x):
    p_x = (x[1:, :]/x[:-1, :])-1
    return global_normalize(p_x)


def load_data(file_name, filter_cols):
    data = pd.read_csv(file_name)
    # data.drop(['Date', 'Volume', 'Market Cap'], axis=1, inplace=True)
    data.drop(filter_cols, axis=1, inplace=True)
    return np.array(data[::-1])


def prepare_windows(data, x_window_len, pred_col):
    x_window_data, y_window_data = [], []
    i = 0
    while i + x_window_len + 1 < len(data):
        window_x = data[i: (i + x_window_len)]
        window_y = data[(i + x_window_len + 1), pred_col]
        x_window_data.append(window_x.T)
        y_window_data.append(window_y)
        i += 1

    x_window_data = np.array(x_window_data)
    y_window_data = np.array(y_window_data)
    
    y_window_data = y_window_data.reshape(y_window_data.shape[0],1)
    
    return x_window_data,y_window_data


def build_model(units, activation, loss, optimizer):
    model = Sequential()
    model.add(LSTM(input_dim=units[0],
                   output_dim=units[1],
                   return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units[2],
                   return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units[3]))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation(activation))

    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    return model


def plot_prediction(pred_test, y_test):
    plt.title("Prediction")
    plt.plot(pred_test[:,0], 'r', label="Prediction")# prediction is in red.
    plt.plot(y_test[:,0], 'b', label="Actual")
    plt.legend()
    plt.show()


def train(model, X_train, Y_train, epochs, batch_size, model_dest):
    print('=========Training started')
    model.fit(X_train, Y_train, batch_size=batch_size,
              epochs=epochs,
              validation_split=0.05)
    print('===========Training finished')
    model.save(model_dest)
    print('===========Model Saved')


if __name__ == "__main__":
    # Configurable parameters
    xw_len = 5 # Parameter for LSTM window length
    pred_col = 3 # Prediction column index
    train_split = 0.85

    data = load_data("input\\bitcoin_day.csv", ['Date', 'Volume', 'Market Cap'])

    # normalize data with column wise maximum
    data, norm_base = global_normalize(data)

    # convert data into sliding window frames
    x_windows, y_windows = prepare_windows(data, xw_len, pred_col)

    # Train data split
    train_x, test_x = x_windows[:int(train_split*x_windows.shape[0]),:,:], x_windows[int(train_split*x_windows.shape[0]):,:,:]
    train_y, test_y = y_windows[:int(train_split*y_windows.shape[0]),:], y_windows[int(train_split*y_windows.shape[0]):,:]

    # building model
    model = build_model([xw_len, xw_len, 50, 100], "sigmoid", "mse", "adam")
    train(model, train_x, train_y, epochs=50, batch_size=50, model_dest="models/lstm_h3_w5.h5")

    # prediction
    pred_y = model.predict(test_x)

    # de-normalizing predictions back to original scale
    pred_y = pred_y * norm_base[pred_col]
    test_y = test_y * norm_base[pred_col]

    print("Model accuracy (whole):", model.evaluate(test_x, test_y))
    print("Model accuracy (sliced):", model.evaluate(test_x[:-100,:,:], test_y[:-100,:]))
    plot_prediction(pred_y, test_y)

    # Predicting training data
    pred_y = model.predict(train_x)
    pred_y = pred_y * norm_base[pred_col]
    test_y = train_y * norm_base[pred_col]
    plot_prediction(pred_y, test_y)




