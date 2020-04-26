from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import utils
import pandas as pd
import numpy as np

model = Sequential()
model.add(Dense(32, activation='tanh', input_shape=(4,)))
model.add(Dense(32, activation='tanh'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Iris dataset process
a = pd.read_csv('iris.csv').to_numpy()

x = a[0:, 0:4]

y = a[0:, 4:]

j = 0
for i in y:
    if i == 'Setosa':
        y[j][0] = 0
    elif i == 'Versicolor':
        y[j][0] = 1
    else:
        y[j][0] = 2

    j += 1

y = utils.to_categorical(y, 3)

model.fit(x, y, epochs=100)

model.save('iris-weight.h5')

