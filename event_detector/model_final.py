import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class EventDetector(keras.Model):
    def __init__(self):
        super(EventDetector, self).__init__()
        self.conv1 = layers.Conv1D(64, 3, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1))
        self.bn1 = layers.BatchNormalization()
        self.mp1 = layers.MaxPool1D(3)

        self.conv2 = layers.Conv1D(128, 3, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1))
        self.bn2 = layers.BatchNormalization()
        self.mp2 = layers.MaxPool1D(3)

        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.15)

        self.d1 = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1))
        self.d2 = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1))
        self.d3 = layers.Dense(3, activation='relu', kernel_regularizer=keras.regularizers.L2(0.1))

        self.softmax = layers.Softmax()

    def call(self, inputs, training):
        res = self.mp1(self.bn1(self.conv1(inputs)))
        res = self.mp2(self.bn2(self.conv2(res)))
        res = self.flatten(res)

        if training:
            res = self.dropout(res)
        
        res = self.d3(self.d2(self.d1(res)))
        res = self.softmax(res)

        return res

    def model(self, shape):
        input = layers.Input(shape=shape)
        return keras.Model(inputs=[input], outputs=self.call(input, False))