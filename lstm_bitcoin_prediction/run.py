#Load libs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt

def load_data(file_name):
    data_raw = pd.read_csv(file_name)
    #Reverse data to oldest to newest
    data_raw = data_raw.iloc[::-1]
    # Extracting features
    data = data_raw[["Open", "High", "Low", "Close"]].values
    print("Data shape: {}".format(data.shape))
    return data

def get_data(data):
    """
    Get training and testing split from the raw data
    """
    data_train, data_test = data[:int(data.shape[0] * 0.8), :], data[int(data.shape[0] * 0.8):, :]
    print('Training data shape: {}'.format(data_train.shape))
    print('Testing data shape: {}'.format(data_test.shape))

    norm_price_variation_train = (1 - (data_train[:, 0] / data_train[:, 3])) * 100
    # norm_highs_train = (data_train[:, 1] / np.maximum(data_train[:, 0], data_train[:, 3]) - 1) * 100
    # norm_low_train = (data_train[:, 2] / np.minimum(data_train[:, 0], data_train[:, 3]) - 1) * 100
    norm_price_variation_test = (1 - (data_test[:, 0] / data_test[:, 3])) * 100
    # norm_highs_test = (data_test[:, 1] / np.maximum(data_test[:, 0], data_test[:, 3]) - 1) * 100
    # norm_low_test = (data_test[:, 2] / np.minimum(data_test[:, 0], data_test[:, 3]) - 1) * 100
    X_train = np.array([norm_price_variation_train]).transpose()
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    Y_train = np.array(np.sign(data_train[1:, 3] / data_train[:-1, 3] - 1))
    X_test = np.array([norm_price_variation_test]).transpose()
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    Y_test = np.array(np.sign(data_test[1:, 3] / data_test[:-1, 3] - 1))
    print('X_train data shape: {}'.format(X_train.shape))
    print('Y_train data shape: {}'.format(Y_train.shape))
    print('X_test data shape: {}'.format(X_test.shape))
    print('Y_test data shape: {}'.format(Y_test.shape))
    return(X_train, X_test, Y_train, Y_test)

def build_model(units, activation, loss, optimizer):
    model = Sequential()
    model.add(LSTM(units[0],
                   input_shape=(None, 1),
                   return_sequences=True
                   ))
    model.add(Dropout(0.1))
    model.add(LSTM(units[1], return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units[2]))
    model.add(Dense(1))
    model.add(Activation(activation))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def train(model, X_train, Y_train, epochs, batch_size):
    print('Training started')
    model.fit(X_train[:-1], Y_train, batch_size=batch_size,
              epochs=epochs,
              validation_split=0.05)
    print('Training finished')

def run():
    # Load train and test data
    X_train, X_test, Y_train, Y_test = get_data(load_data('./input/bitcoin_day.csv'))
    #build the LSTM mdoel
    model = build_model(units=[100, 100, 50], activation='tanh', loss='mse', optimizer='adam')
    train(model, X_train, Y_train, epochs=500, batch_size=50)
    pred = model.predict(X_test)
    predicted = np.sign(pred)
    for p in zip(predicted,Y_test):
        print('Predicted, Y_test {}:'.format(p))
    #lets plot the last predictions in comparison to the actual variations
    plt.plot(predicted[-50:],'r')#prediction is in red.
    # plt.plot(data_test[:], 'b')#actual in blue.
    plt.plot(Y_test[-50:],'b')
    plt.show()

run()