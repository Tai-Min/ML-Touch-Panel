import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from model import SiamFC
import numpy as np
import cv2

# get inputs to nnet
exemplar = tf.io.read_file('D:/dli/pointer_tracker/exemplar.JPEG')
exemplar = tf.cast(tf.io.decode_jpeg(exemplar), tf.float32)/255
exemplar = tf.expand_dims(exemplar, 0)
src_frame = tf.io.read_file('./source_frame.JPEG')
src_frame = tf.cast(tf.io.decode_jpeg(src_frame), tf.float32)/255
src_frame = tf.expand_dims(src_frame, 0)
inputs = [exemplar, src_frame]

# create nnet model
model = SiamFC()
model.model((None, None, 3), (None, None, 3)).summary()

# restore weights from checkpoint
ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=1000)

path = manager.restore_or_initialize()
if path:
    print('Restored checkpoint from %s' % path)
else:
    print('No checkpoints found.')

# predict score map
preds = model(inputs)
preds = tf.squeeze(preds, 0)

# get original image as matrix and also additional matrix for final result
src_frame = cv2.imread('./source_frame.JPEG')
src_frame_final = np.zeros_like(src_frame)

# convert image to grayscale with three channels
src_frame = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
src_frame_final[:,:,0] = src_frame
src_frame_final[:,:,1] = src_frame
src_frame_final[:,:,2] = src_frame

# cast predictions to uint8 and resize them to image's size
preds = preds.numpy()
preds = cv2.resize(preds, (src_frame_final.shape[1], src_frame_final.shape[0]))

# find coords of biggest prediction to draw circle over it
i,j = np.unravel_index(preds.argmax(), preds.shape)

# generate heatmap from predictions and place it over grayscale image
heatmapshow  = None
heatmapshow  = cv2.normalize(preds, heatmapshow , alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
preds  = cv2.applyColorMap(heatmapshow , cv2.COLORMAP_JET)
src_frame_final = cv2.addWeighted(src_frame_final, 0.6, preds, 0.4, 0.0)

# draw circle over most confident place in score map
src_frame_final = cv2.circle(src_frame_final, (j,i), 10, (255, 255, 255), -1)

# show the result
cv2.imshow('detection', src_frame_final)
cv2.waitKey(0)