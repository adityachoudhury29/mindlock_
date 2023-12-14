import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Reshape, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('./sudoku.csv')
df_subset = df.sample(n=500000, random_state=42)

X = np.array(df.quizzes.map(lambda x: list(map(int, x))).to_list())
Y = np.array(df.solutions.map(lambda x: list(map(int, x))).to_list())

X = X.reshape(-1, 9, 9, 1)
Y = Y.reshape(-1, 9, 9) - 1

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

model = Sequential()
model.add(Conv2D(128, 3, activation='relu', padding='same', input_shape=(9,9,1)))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(1024, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(9, 1, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(81*9))
model.add(tf.keras.layers.LayerNormalization(axis=-1))
model.add(Reshape((9, 9, 9)))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
                optimizer=Adam(
                learning_rate=0.001
    ),
    metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, batch_size = 64, epochs = 15,validation_data=(X_test, y_test))

model.evaluate(X_test, y_test)

model.save('sudokumodel-gpu-3.h5')
