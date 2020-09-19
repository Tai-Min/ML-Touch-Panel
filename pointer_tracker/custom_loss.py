import tensorflow as tf
import scipy.signal
import numpy as np

# get radius from center of the image using Pythagorean Theorem
# this function accounts for the stride of the network 
# so returned radius is in reference to source frame and not score map
@tf.function
def get_radius_from_center(x, y, center, stride):
    x_with_offset = tf.math.add(float(x),0.5)
    y_with_offset = tf.math.add(float(y),0.5)
    x_distance = tf.math.subtract(tf.cast(center, tf.float32), x_with_offset)
    y_distance = tf.math.subtract(tf.cast(center, tf.float32), y_with_offset)
    x_vec_part = tf.math.pow(x_distance, 2)
    y_vec_part = tf.math.pow(y_distance, 2)
    distance = tf.math.sqrt(tf.math.add(x_vec_part, y_vec_part))
    
    # account for stride of the network
    return tf.math.multiply(tf.cast(stride, tf.float32), distance)

# return cosine matrix of given shape
@tf.function
def get_cosine_matrix(shape):
    cos0 = tf.convert_to_tensor(np.tile(scipy.signal.cosine(shape[1]), (shape[0], 1)), tf.float32)
    cos1 = tf.convert_to_tensor(np.tile(np.transpose([scipy.signal.cosine(shape[0])]), (1,shape[1])), tf.float32)
    return tf.math.multiply(cos0, cos1)

# get map of logistic losses
# for every cell in prediction map
@tf.function
def get_loss_map(x):
    sample = x[0]
    pred = tf.divide(x[1], 1000) # change from original paper to prevent inf on tf.math.exp
    max_radius = x[2]
    weight = x[3]
    
    stride = tf.math.divide(sample.shape[0], pred.shape[0])
    center = tf.math.divide(pred.shape[0], 2)

    shapes = [pred.shape[0], pred.shape[1]]

    # construct raw loss map -1 if too far from center, 1 otherwise
    loss_map =  tf.convert_to_tensor([
                        [
                                1.0 if tf.math.less(get_radius_from_center(j, i, center, stride), max_radius)
                                else -1.0 
                            for j in range(shapes[1])
                        ]
                        for i in range(shapes[0])
                    ], dtype=tf.float32)

    # add cosine window to punish examples that are too far
    # and reward those closer
    loss_map = tf.math.add(loss_map, get_cosine_matrix(shapes)),

    # apply weight to make sure that all video folders are treated equally
    # and the network won't learn filters based on one dominating video folder
    loss_map = tf.math.multiply(weight, loss_map)

    # get logistic loss
    loss_map = tf.math.multiply(loss_map, pred)
    loss_map = tf.multiply(-1.0, loss_map)
    loss_map = tf.math.exp(loss_map)
    loss_map = tf.math.add(1.0, loss_map)
    loss_map = tf.math.log(loss_map)
    return tf.squeeze(loss_map, 0)

@tf.function
def logistic_loss_fn(samples, preds, radiuses, weights):
    source_frames = samples[1] # only source frames are required to compute loss
    radiuses = radiuses[1] # only radiuses of source frames are required to compute loss

    losses = tf.map_fn(
        fn = get_loss_map,
        elems=(source_frames, preds, radiuses, weights),
        fn_output_signature=tf.TensorSpec([None, None])
    )

    return tf.math.reduce_mean(losses), losses
