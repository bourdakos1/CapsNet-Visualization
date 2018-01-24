from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow as tf
import numpy as np
import os

PATH_TO_CKPT = 'tensorflow_checkpoint'
MODEL_VERSION = 'model_epoch_0047_step_20591'
PATH_TO_MODEL = os.path.join(PATH_TO_CKPT, MODEL_VERSION)

PATH_TO_WEIGHTS = 'numpy_weights'
PATH_TO_CONV1 = os.path.join(PATH_TO_WEIGHTS, 'conv1.weights.npz')
PATH_TO_CONV1_BIAS = os.path.join(PATH_TO_WEIGHTS, 'conv1.bias.npz')
PATH_TO_PRIMARY_CAPS = os.path.join(PATH_TO_WEIGHTS, 'primary_caps.weights.npz')
PATH_TO_PRIMARY_CAPS_BIAS = os.path.join(PATH_TO_WEIGHTS, 'primary_caps.bias.npz')
PATH_TO_DIGIT_CAPS = os.path.join(PATH_TO_WEIGHTS, 'digit_caps.weights.npz')
PATH_TO_FULLY_CONNECTED1 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected1.weights.npz')
PATH_TO_FULLY_CONNECTED2 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected2.weights.npz')
PATH_TO_FULLY_CONNECTED3 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected3.weights.npz')

print_tensors_in_checkpoint_file(file_name=PATH_TO_MODEL, tensor_name='', all_tensors=False)

sess = tf.Session()
new_saver = tf.train.import_meta_graph(PATH_TO_MODEL + '.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(PATH_TO_CKPT))

# Conv1_layer/Conv/weights (DT_FLOAT) [9,9,1,256]
weights = sess.run('Conv1_layer/Conv/weights:0')
with open(PATH_TO_CONV1, 'wb') as outfile:
    np.save(outfile, weights)

# Conv1_layer/Conv/biases (DT_FLOAT) [256]
bias = sess.run('Conv1_layer/Conv/biases:0')
with open(PATH_TO_CONV1_BIAS, 'wb') as outfile:
    np.save(outfile, bias)

# PrimaryCaps_layer/Conv/weights (DT_FLOAT) [9,9,256,256]
weights = sess.run('PrimaryCaps_layer/Conv/weights:0')
with open(PATH_TO_PRIMARY_CAPS, 'wb') as outfile:
    np.save(outfile, weights)

# PrimaryCaps_layer/Conv/biases (DT_FLOAT) [256]
bias = sess.run('PrimaryCaps_layer/Conv/biases:0')
with open(PATH_TO_PRIMARY_CAPS_BIAS, 'wb') as outfile:
    np.save(outfile, bias)

# DigitCaps_layer/routing/Weight (DT_FLOAT) [1,1152,10,8,16]
weights = sess.run('DigitCaps_layer/routing/Weight:0')
with open(PATH_TO_DIGIT_CAPS, 'wb') as outfile:
    np.save(outfile, weights)

# Decoder/fully_connected/weights (DT_FLOAT) [16,512]
weights = sess.run('Decoder/fully_connected/weights:0')
with open(PATH_TO_FULLY_CONNECTED1, 'wb') as outfile:
    np.save(outfile, weights)

# Decoder/fully_connected_1/weights (DT_FLOAT) [512,1024]
weights = sess.run('Decoder/fully_connected_1/weights:0')
with open(PATH_TO_FULLY_CONNECTED2, 'wb') as outfile:
    np.save(outfile, weights)

# Decoder/fully_connected_2/weights (DT_FLOAT) [1024,784]
weights = sess.run('Decoder/fully_connected_2/weights:0')
with open(PATH_TO_FULLY_CONNECTED3, 'wb') as outfile:
    np.save(outfile, weights)
