import tensorflow as tf
import os
import shutil

def export_model(model, dir, filename):
    model.save('model.tf')
    print(filename)
    os.system('%%INTEL_OPENVINO_DIR%%/deployment_tools/model_optimizer/mo.py --saved_model_dir model.tf -b 1 --output_dir %s --model_name %s' % (dir, filename))
    shutil.rmtree('model.tf')