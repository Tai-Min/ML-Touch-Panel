import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SiameseConvEmbedding(layers.Layer):
    def __init__(self):
        super(SiameseConvEmbedding, self).__init__()
        self.conv1 = layers.Conv2D(96, 11, strides=2, padding='valid')
        self.bn1 = layers.BatchNormalization(trainable=False)
        self.relu1 = layers.ReLU()
        
        self.pool1 = layers.MaxPool2D(pool_size=3, strides=2, padding='valid')

        self.conv2 = layers.Conv2D(256, 5, strides=1, padding='valid')
        self.bn2 = layers.BatchNormalization(trainable=False)
        self.relu2 = layers.ReLU()

        self.pool2 = layers.MaxPool2D(pool_size=3, strides=2, padding='valid')

        self.conv3 = layers.Conv2D(384, 3, strides=1, padding='valid')
        self.bn3 = layers.BatchNormalization(trainable=False)
        self.relu3 = layers.ReLU()

        self.conv4 = layers.Conv2D(384, 3, strides=1, padding='valid')
        self.bn4 = layers.BatchNormalization(trainable=False)
        self.relu4 = layers.ReLU()

        self.conv5 = layers.Conv2D(256, 3, strides=1, padding='valid')
        self.bn5 = layers.BatchNormalization(trainable=False)

    def call(self, inputs, training = None):
        res = self.conv1(inputs)
        if training:
            res = self.bn1(res)
        res = self.relu1(res)

        res = self.pool1(res)

        res = self.conv2(res)
        if training:
            res = self.bn2(res)
        res = self.relu2(res)

        res = self.pool2(res)

        res = self.conv3(res)
        if training:
            res = self.bn3(res)
        res = self.relu3(res)

        res = self.conv4(res)
        if training:
            res = self.bn4(res)
        res = self.relu4(res)

        res = self.conv5(res)
        if training:
            res = self.bn5(res)

        return res

class SiamFC(keras.Model):
    def __init__(self):
        super(SiamFC, self).__init__()
        self.embedding = SiameseConvEmbedding()

    def convSingle(self, x):
        return tf.squeeze(tf.nn.conv2d(tf.expand_dims(x[0], 0), tf.expand_dims(x[1], -1), [1,1,1,1], 'VALID'), 0)

    def call(self, inputs, training = None):

        exemplar = inputs[0]
        source_frame = inputs[1]
    
        filters = self.embedding(exemplar, training)
        inputs = self.embedding(source_frame, training)

        # corss corelation layer
        result = tf.map_fn(
            fn = self.convSingle,
            elems=(inputs, filters),
            fn_output_signature=tf.TensorSpec([None, None, 1])
        )   

        return tf.squeeze(result, -1)

    def model(self, ex_shape, src_shape):
        exp = layers.Input(ex_shape, batch_size=1) 
        src = layers.Input(src_shape, batch_size=1)
        return keras.Model(inputs=[exp, src], outputs=self.call([exp, src], False))
