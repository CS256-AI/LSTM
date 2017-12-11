#Load libs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt

#Lets load the daily prices
btc = pd.read_csv('./input/bitcoin_day.csv')
# print(btc.head())

#data is in reverse order
btc = btc.iloc[::-1]
# print(btc.head())

#getting the 4 price-related features from the dataframe
data = btc[["Open", "High", "Low", "Close"]].values
print("Data shape: {}".format(data.shape))
data_train = data[:int(data.shape[0] * 0.8), :]
data_test = data[int(data.shape[0] * 0.8):, :]
print('Training data shape: {}'.format(data_train.shape))
print('Testing data shape: {}'.format(data_test.shape))

#we change the data to have something more generalizeable, lets say [ %variation , %high, %low]
norm_price_variation_train = (1 - (data_train[:, 0] / data_train[:, 3])) * 100
norm_highs_train = (data_train[:, 1] / np.maximum(data_train[:, 0], data_train[:, 3]) - 1) * 100
norm_low_train = (data_train[:, 2] / np.minimum(data_train[:, 0], data_train[:, 3]) - 1) * 100

X_train = np.array([norm_price_variation_train , norm_highs_train, norm_low_train]).transpose()
#little trick to make X_train a 3 dimensional array for LSTM input shape
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

print(X_train[:2])

#We generate Y_train. For this update, we will only determine if the trend is up or down for 2 days ahead
Y_train = np.array((np.sign((data_train[2:, 3] / data_train[:-2, 3] - 1)) + 1) / 2)
print(Y_train[:10])

#Lets make a simple lstm model
#I got it from online tutorial
model = Sequential()
model.add(LSTM(100,
               input_shape = (None,1),
               return_sequences = True
              ))
model.add(Dropout(0.1))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(50))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print('Training started')
model.compile(loss="mse", optimizer="rmsprop")
model.fit(X_train[:-2],Y_train, batch_size=512,
	    epochs=500,
	    validation_split=0.05)
print('Training finished')

norm_price_variation_test = (1 - (data_test[:, 0] / data_test[:, 3])) * 100
norm_highs_test = (data_test[:, 1] / np.maximum(data_test[:, 0], data_test[:, 3]) - 1) * 100
norm_low_test = (data_test[:, 2] / np.minimum(data_test[:, 0], data_test[:, 3]) - 1) * 100

X_test = np.array([norm_price_variation_test , norm_highs_test, norm_low_test]).transpose()
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
Y_test = np.array((np.sign(data_test[2:, 3] / data_test[:-2, 3] - 1) + 1) / 2)
print('X_test data shape: {}'.format(X_test.shape))
print('Y_test data shape: {}'.format(Y_test.shape))

# model.evaluate(X_test[:-2],Y_test)

pred = model.predict(X_test)

predicted = (np.sign(pred-0.45)+1)/2*50

for p in zip(predicted,Y_test):
    print('Predicted, Y_test {}:'.format(p))
#lets plot the last predictions in comparison to the actual variations
plt.plot(predicted[:],'r')#prediction is in red.
# plt.plot(data_test[:], 'b')#actual in blue.
plt.plot(Y_test*50,'b')
plt.show()