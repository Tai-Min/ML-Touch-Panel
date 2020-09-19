import tensorflow as tf
from tensorflow import keras
from dataset_loader import dataset_loader
from model import SiamFC
from custom_loss import logistic_loss_fn

train_radiuses_path = './dataset_new/Annotations/train'
train_data_path = './dataset_new/Data/train'

epochs = 50
train_samples_per_epoch = 50000
batch_size = 8
init_lr = 0.01
final_lr = 0.00001

# get steps and decay rate for exponential decay to go from init_lr to final lr
# in given number of epochs
train_batches_per_epoch = train_samples_per_epoch / batch_size
lr_decay_steps = train_batches_per_epoch * epochs
lr_decay_rate = final_lr / init_lr

train_dataset = tf.data.Dataset.from_generator(dataset_loader, ((tf.string, tf.string), (tf.string, tf.string), tf.float32), args=(train_data_path, train_radiuses_path))

# load images and labels received from generator
@tf.function
def load_data(data_paths, radius_paths, weights):
        exemplar_data = tf.io.read_file(data_paths[0])
        exemplar_radius = tf.io.read_file(radius_paths[0])
        frame_data = tf.io.read_file(data_paths[1])
        frame_radius = tf.io.read_file(radius_paths[1])

        exemplar_data_decoded = tf.math.divide(tf.cast(tf.io.decode_jpeg(exemplar_data), tf.float32), 255)
        exemplar_radius_decoded = tf.strings.to_number(exemplar_radius)
        frame_data_decoded = tf.math.divide(tf.cast(tf.io.decode_jpeg(frame_data), tf.float32), 255)
        frame_radius_decoded = tf.strings.to_number(frame_radius)

        return ((exemplar_data_decoded, frame_data_decoded), (exemplar_radius_decoded, frame_radius_decoded), tf.cast(weights, tf.float32))

# load images outside of generator to relieve it a bit
train_dataset = train_dataset.map(load_data)

# batch and prefetch dataset
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

model = SiamFC()
model.model((None, None, 3), (None, None, 3)).summary()

lr_schedule = keras.optimizers.schedules.ExponentialDecay(init_lr, decay_steps=lr_decay_steps, decay_rate=lr_decay_rate, staircase=False)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=1000)

train_loss_metric = keras.metrics.Mean()

# restore saved session
path = manager.restore_or_initialize()
if path:
    print('Restored checkpoint from %s' % path)
else:
    print('Initializing training from scratch.')

@tf.function
def train_step(inputs, radiuses, weights):
    with tf.GradientTape() as tape:
        preds = model(inputs, True)
        losses, loss_maps = logistic_loss_fn(inputs, preds, radiuses, weights)
        
    if tf.math.is_finite(losses):
        grads = tape.gradient(losses, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_loss_metric(losses)

for epoch in range(epochs):
    for _ in range(int(train_batches_per_epoch)):
        sample_batch, radius_batch, weights_batch = next(iter(train_dataset))
        train_step(sample_batch, radius_batch, weights_batch)
        
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 200 == 0:
            path = manager.save()
            print("Checkpoint saved: %s." % path)
            
        if int(ckpt.step) % 2000 == 0:
            model.model((None, None, 3), (None, None, 3)).save('model.tf')
            print("Model model.tf saved.")

        print("Mean loss on training batch: %s." % (float(train_loss_metric.result())))
        train_loss_metric.reset_states()