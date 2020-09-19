# cross corelation layer is not yet supported by OpenVINO
# so cross corelation must be done in c++ code instead
# so this program only exports embedding functions to IR model

source_height = int(720/4)
source_width = int(source_height*16/9)
exemplar_height = int(720/8)
exemplar_width = int(exemplar_height*16/9)

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import shutil
from model import SiamFC

class EmbeddingModel(keras.Model):
    def __init__(self, embedding):
        super(EmbeddingModel, self).__init__()
        self.embedding = embedding

    def call(self, input, training = None):   
        return self.embedding(input, training)

    def model(self, shape):
        input = layers.Input(shape, batch_size=1, name='input')
        return keras.Model(inputs=[input], outputs=self.call(input, False))

def export_model(model, dir, filename):
    model.save('model.tf')
    os.system('%%INTEL_OPENVINO_DIR%%/deployment_tools/model_optimizer/mo.py --reverse_input_channels --saved_model_dir model.tf --data_type FP16 --scale 255 --output_dir %s --model_name %s' % (dir, filename))
    shutil.rmtree('model.tf')

# load original model
model = SiamFC()
model.model((exemplar_height, exemplar_width, 3), (source_height, source_width, 3)).summary()

ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=1000)

path = manager.restore_or_initialize()
if path:
    print('Restored checkpoint from %s' % path)
else:
    print('No checkpoints found.')

# export embeddings as separate networks
embedding_src = EmbeddingModel(model.embedding)
embedding_src.model((source_height, source_width, 3)).summary()
export_model(embedding_src.model((source_height, source_width, 3)), 'IR_model', 'embedding_source_subnet')

embedding_exe = EmbeddingModel(model.embedding)
embedding_exe.model((exemplar_height, exemplar_width, 3)).summary()
export_model(embedding_exe.model((exemplar_height, exemplar_width, 3)), 'IR_model', 'embedding_exemplar_subnet')