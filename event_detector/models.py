import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model_cnn_conv1d = keras.Sequential([
    layers.Conv1D(64, 3, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
    layers.BatchNormalization(),
    layers.MaxPool1D(3),
    layers.Conv1D(128, 3, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
    layers.BatchNormalization(),
    layers.MaxPool1D(3),
    layers.Flatten(),
    layers.Dropout(0.15),
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
    layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
    layers.Dense(3, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
    layers.Softmax()
])

model_crnn_conv1d = keras.Sequential([
    layers.Conv1D(64, 3, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
    layers.BatchNormalization(),
    layers.MaxPool1D(3),
    layers.Dropout(0.15),
    layers.GRU(24),
    layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
    layers.Dense(3, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
    layers.Softmax()
])

model_cnn_conv2d = keras.Sequential([
    layers.Conv2D(64, 3, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
    layers.Conv2D(128, 3, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
    layers.MaxPool2D(3),
    layers.Flatten(),
    layers.Dropout(0.15),
    layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
    layers.Dropout(0.15),
    layers.Dense(3, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1)),
    layers.Softmax()
])