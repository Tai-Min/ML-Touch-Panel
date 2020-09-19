import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import dataset_loader as dl
from models import model_cnn_conv1d, model_cnn_conv2d, model_crnn_conv1d
from model_final import EventDetector
from model_exporter import export_model

conv_type = '1d'
model_to_use = 'cnn'
final = True

dataset_path = './dataset'

train_dataset_path = os.path.join(dataset_path, 'train')
test_dataset_path = os.path.join(dataset_path, 'test')

train_dataset_raw = dl.load_dataset(train_dataset_path)
test_dataset_raw = dl.load_dataset(test_dataset_path)

labels_dict = train_dataset_raw['labels_str']
labels_count = len(labels_dict)

train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset_raw['features'], train_dataset_raw['labels']))
test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset_raw['features'], test_dataset_raw['labels']))

if conv_type == '2d':
    train_dataset = train_dataset.map(lambda x, y: (tf.expand_dims(x, -1), tf.keras.backend.one_hot(y, labels_count)))
    test_dataset = test_dataset.map(lambda x, y: (tf.expand_dims(x, -1), tf.keras.backend.one_hot(y, labels_count)))
else:
    train_dataset = train_dataset.map(lambda x, y: (x, tf.keras.backend.one_hot(y, labels_count)))
    test_dataset = test_dataset.map(lambda x, y: (x, tf.keras.backend.one_hot(y, labels_count)))

SHUFFLE_BUFFER_SIZE = len(train_dataset_raw['features'])
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE)

train_dataset = train_dataset.skip(300)
valid_dataset = train_dataset.take(300)

BATCH_SIZE = 128
train_dataset = train_dataset.batch(BATCH_SIZE)
valid_dataset = valid_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

if final:
    model = EventDetector()
elif model_to_use == 'cnn' and conv_type == '1d':
    model = model_cnn_conv1d
    model.build((None,) + train_dataset_raw['features'][0].shape)
elif model_to_use == 'cnn' and conv_type == '2d':
    model = model_cnn_conv2d
    model.build((None,) + train_dataset_raw['features'][0].shape + (1,))
elif model_to_use == 'crnn' and conv_type == '1d':
    model = model_crnn_conv1d
    model.build((None,) + train_dataset_raw['features'][0].shape)
else:
    raise ValueError('Can\'t create CRNN with 2d conv layers.')

if final:
    if conv_type == '1d':
        model.model(train_dataset_raw['features'][0].shape).summary()
    else:
        model.model(train_dataset_raw['features'][0].shape + (1,)).summary()
else:
    model.summary()

optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

train_loss_metric = tf.keras.metrics.Mean()
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_loss_metric = tf.keras.metrics.Mean()
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

epochs = 100

@tf.function
def train_step(x,y):
    with tf.GradientTape() as tape:
        outputs = model(x, training=True)
        loss = loss_fn(y, outputs)
        loss += sum(model.losses)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_loss_metric(loss)
    train_acc_metric(y, outputs)

@tf.function
def valid_step(x,y):
    outputs = model(x, training=False)
    loss = loss_fn(y, outputs)
    test_loss_metric(loss)
    test_acc_metric(y, outputs)

for epoch in range(epochs):
    train_loss_metric.reset_states()
    train_acc_metric.reset_states()
    test_loss_metric.reset_states()
    test_acc_metric.reset_states()

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    
    # train
    for x_train_batch, y_train_batch in train_dataset:
        train_step(x_train_batch, y_train_batch)
    print('Epoch %d: mean loss = %.4f, categorical accuracy = %.4f' % (epoch+1, train_loss_metric.result(), train_acc_metric.result()))

    # validate
    for x_valid_batch, y_valid_batch in valid_dataset:
        valid_step(x_valid_batch, y_valid_batch)
    print('Validation %d: mean loss = %.4f, categorical accuracy = %.4f' % (epoch+1, test_loss_metric.result(), test_acc_metric.result()))

test_loss_metric.reset_states()
test_acc_metric.reset_states()

for x_test_batch, y_test_batch in test_dataset:
    valid_step(x_test_batch, y_test_batch)
print('Testing: mean loss = %.4f, categorical accuracy = %.4f' % (test_loss_metric.result(), test_acc_metric.result()))

acc = str(int(test_acc_metric.result().numpy() * 100))
export_model(model, './IR_model', 'model_' + model_to_use + '_' + conv_type + '_acc_' + acc + '.xml')