import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as image
from scipy import signal
from matplotlib.colors import LinearSegmentedColormap
import json

PATH_TO_TEST_IMAGES_DIR = 'test_images'
PATH_TO_TEST_IMAGE = sys.argv[1]

# Black and white color map going from (0, 0, 0) "black" to (1, 1, 1) "white".
CMAP = LinearSegmentedColormap.from_list('greyscale', ((0, 0, 0), (1, 1, 1)), N=256, gamma=1.0)

# Output directories for visualizations.
PATH_TO_ROOT = 'visualizations'
PATH_TO_VISUALIZATIONS = os.path.join(PATH_TO_ROOT, PATH_TO_TEST_IMAGE)
if not os.path.exists(PATH_TO_VISUALIZATIONS):
    os.mkdir(PATH_TO_VISUALIZATIONS)

PATH_TO_INPUT_IMAGE = os.path.join(PATH_TO_VISUALIZATIONS, '0')
PATH_TO_CONV1_KERNEL = os.path.join(PATH_TO_VISUALIZATIONS, '1')
PATH_TO_RELU = os.path.join(PATH_TO_VISUALIZATIONS, '2')
PATH_TO_PRIMARY_CAPS = os.path.join(PATH_TO_VISUALIZATIONS, '3')
PATH_TO_DIGIT_CAPS = os.path.join(PATH_TO_VISUALIZATIONS, '4')
PATH_TO_RECONSTRUCTION = os.path.join(PATH_TO_VISUALIZATIONS, '5')
PATH_TO_RECONSTRUCTION_JSON_PARAMS = os.path.join(PATH_TO_VISUALIZATIONS, '6')

if not os.path.exists(PATH_TO_INPUT_IMAGE):
    os.mkdir(PATH_TO_INPUT_IMAGE)
if not os.path.exists(PATH_TO_CONV1_KERNEL):
    os.mkdir(PATH_TO_CONV1_KERNEL)
if not os.path.exists(PATH_TO_RELU):
    os.mkdir(PATH_TO_RELU)
if not os.path.exists(PATH_TO_PRIMARY_CAPS):
    os.mkdir(PATH_TO_PRIMARY_CAPS)
if not os.path.exists(PATH_TO_DIGIT_CAPS):
    os.mkdir(PATH_TO_DIGIT_CAPS)
if not os.path.exists(PATH_TO_RECONSTRUCTION):
    os.mkdir(PATH_TO_RECONSTRUCTION)
if not os.path.exists(PATH_TO_RECONSTRUCTION_JSON_PARAMS):
    os.mkdir(PATH_TO_RECONSTRUCTION_JSON_PARAMS)

# Input directories for layer weights.
PATH_TO_WEIGHTS = 'numpy_weights'
PATH_TO_WEIGHTS_CONV1 = os.path.join(PATH_TO_WEIGHTS, 'conv1.weights.npz')
PATH_TO_WEIGHTS_CONV1_BIAS = os.path.join(PATH_TO_WEIGHTS, 'conv1.bias.npz')
PATH_TO_WEIGHTS_PRIMARY_CAPS = os.path.join(PATH_TO_WEIGHTS, 'primary_caps.weights.npz')
PATH_TO_WEIGHTS_PRIMARY_CAPS_BIAS = os.path.join(PATH_TO_WEIGHTS, 'primary_caps.bias.npz')
PATH_TO_WEIGHTS_DIGIT_CAPS = os.path.join(PATH_TO_WEIGHTS, 'digit_caps.weights.npz')
PATH_TO_WEIGHTS_FULLY_CONNECTED1 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected1.weights.npz')
PATH_TO_WEIGHTS_FULLY_CONNECTED2 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected2.weights.npz')
PATH_TO_WEIGHTS_FULLY_CONNECTED3 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected3.weights.npz')
PATH_TO_WEIGHTS_FULLY_CONNECTED1_BIAS = os.path.join(PATH_TO_WEIGHTS, 'fully_connected1.bias.npz')
PATH_TO_WEIGHTS_FULLY_CONNECTED2_BIAS = os.path.join(PATH_TO_WEIGHTS, 'fully_connected2.bias.npz')
PATH_TO_WEIGHTS_FULLY_CONNECTED3_BIAS = os.path.join(PATH_TO_WEIGHTS, 'fully_connected3.bias.npz')

# Number of routing iterations to do in DigitCaps layer.
NUMBER_OF_ROUNDS = 3

# Load the weights with numpy.
conv1_weights = np.load(PATH_TO_WEIGHTS_CONV1)
conv1_bias = np.load(PATH_TO_WEIGHTS_CONV1_BIAS)
primary_caps_weights = np.load(PATH_TO_WEIGHTS_PRIMARY_CAPS)
primary_caps_bias = np.load(PATH_TO_WEIGHTS_PRIMARY_CAPS_BIAS)
digit_caps = np.load(PATH_TO_WEIGHTS_DIGIT_CAPS)
fully_connected1 = np.load(PATH_TO_WEIGHTS_FULLY_CONNECTED1)
fully_connected2 = np.load(PATH_TO_WEIGHTS_FULLY_CONNECTED2)
fully_connected3 = np.load(PATH_TO_WEIGHTS_FULLY_CONNECTED3)
fully_connected1_bias = np.load(PATH_TO_WEIGHTS_FULLY_CONNECTED1_BIAS)
fully_connected2_bias = np.load(PATH_TO_WEIGHTS_FULLY_CONNECTED2_BIAS)
fully_connected3_bias = np.load(PATH_TO_WEIGHTS_FULLY_CONNECTED3_BIAS)

################################################################################
# Helper Functions
################################################################################
def squash(s, axis=-1, epsilon=1e-9):
    squared_norm = np.sum(np.square(s), axis=axis, keepdims=True)
    safe_norm = np.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = s / safe_norm
    return squash_factor * unit_vector


def safe_norm(s, axis=-1, epsilon=1e-9, keepdims=False):
    squared_norm = np.sum(np.square(s), axis=axis, keepdims=keepdims)
    return np.sqrt(squared_norm + epsilon)


expit = lambda x: 1.0 / (1 + np.exp(-x))
def sigmoid_function(signal):
    # Prevent overflow.
    signal = np.clip(signal, -500, 500)
    # Calculate activation signal
    return expit(signal)


def ReLU_function(signal):
    # Return the activation signal
    return np.maximum(0, signal)


################################################################################
# Load Input Image
################################################################################
img = Image.open(os.path.join(PATH_TO_TEST_IMAGES_DIR, PATH_TO_TEST_IMAGE))
# Save original image to the visualization folder
image.imsave(os.path.join(PATH_TO_INPUT_IMAGE, '{}.png'.format(0)), img)
input_image = np.array(img.getdata(), dtype=np.uint8)
input_image = np.resize(input_image, (img.size[1], img.size[0], 4))
input_image = input_image[:, :, 1]


################################################################################
# Convolution 1
################################################################################
conv1_output = np.empty((20, 20, 256))
for i in range(conv1_weights.shape[3]):
    # Get the 9x9 kernel
    extracted_filter = conv1_weights[:, :, :, i]
    extracted_filter = np.squeeze(extracted_filter)

    # Save image of the kernel
    image.imsave(os.path.join(PATH_TO_CONV1_KERNEL, '{}.png'.format(i)), extracted_filter, vmin=-0.6064218, vmax=0.24946211)

    # Apply convolution
    conv1 = signal.correlate2d(input_image, extracted_filter, 'valid')
    conv1 = conv1 + conv1_bias[i]

    # ReLU
    conv1 = np.maximum(0, conv1)
    conv1_output[:, :, i] = conv1

    # Save image of the the convolution and ReLU
    image.imsave(os.path.join(PATH_TO_RELU, '{}.png'.format(i)), conv1, cmap=CMAP, vmin=0, vmax=255)


################################################################################
# PrimaryCaps
################################################################################
primary_caps_output = np.empty((6, 6, 256))
for i in range(primary_caps_weights.shape[3]):
    # Get the 9x9x256 kernel
    extracted_filter = primary_caps_weights[:, :, :, i]

    # Apply convolution
    conv2 = signal.correlate(conv1_output, extracted_filter, 'valid')
    conv2 = conv2 + primary_caps_bias[i]

    # Outputs 12x12, but we need 6x6 so we drop every other item
    conv2 = conv2[::2, ::2, :]
    conv2 = np.squeeze(conv2)

    # ReLU
    conv2 = np.maximum(0, conv2)
    primary_caps_output[:, :, i] = conv2


# The paper says that a PrimaryCaps layer is a 6x6 matrix of 8 dimensional
# vectors and there should be 32 PrimaryCaps layers. Meaning we have 6x6x32 vectors
# which equals 1,152 vectors.
# We only really need a list of all the 8d vectors hence we can reshape the matrix
# to: (1152, 8, 1)
primary_caps_output = np.reshape(primary_caps_output, (-1, 8, 1))
# Squash
squashed_caps_output = squash(primary_caps_output)

# Render the Capsule Layers after squashing
normed_squashed_caps_output = np.reshape(safe_norm(squashed_caps_output, axis=-2), (6, 6, 32))
for i in range(normed_squashed_caps_output.shape[2]):
    image.imsave(os.path.join(PATH_TO_PRIMARY_CAPS, '{}.png'.format(i)), normed_squashed_caps_output[:, :, i], cmap=CMAP, vmin=0, vmax=1)


################################################################################
# DigitCaps and Routing
################################################################################

# Add in a blank dimension: (1, 1152, 1, 8, 1)
squashed_caps_output = np.reshape(squashed_caps_output, (-1, 1, squashed_caps_output.shape[-2], 1))
# Tile the capsules in the inserted dimension: (1, 1152, 10, 8, 1)
caps_output_tiled = np.tile(squashed_caps_output, [1, 1, 10, 1, 1])

# Transpose DigitCaps (1, 1152, 10, 8, 16) -> (1, 1152, 10, 16, 8)
# matmul caps_output_tiled (1152, 10, 8, 1) by digit_caps (1, 1152, 10, 16, 8)
# It's doing a matrix multiplication on the last 2 dimensions:
#                 │ 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 │ 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 │ 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 │ 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
#                 │ 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
#                 │ 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
#                 │ 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
#                 │ 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
# ────────────────┼────────────────────────────────
# 1 2 3 4 5 6 7 8 │ 1 2 3 4 5 6 7 8 0 0 0 0 0 0 0 0

caps2_predicted = np.matmul(np.transpose(digit_caps, (0, 1, 2, 4, 3)), caps_output_tiled)

raw_weights = np.zeros([1, 1152, 10, 1, 1])

# Routing iterations
for x in range(0, NUMBER_OF_ROUNDS):
    # Softmax
    routing_weights = np.exp(raw_weights) / np.sum(np.exp(raw_weights), axis=2, keepdims=True)
    # Sumation of the element wise multiplication
    weighted_predictions = np.multiply(routing_weights, caps2_predicted)
    weighted_sum = np.sum(weighted_predictions, axis=1, keepdims=True)
    # Squash
    caps2_output = squash(weighted_sum, axis=-2)

    # We don't need to recalcute raw weights on the last iteration
    if x < NUMBER_OF_ROUNDS:
        # Add dot product to the weights
        caps2_output_tiled = np.tile(caps2_output, [1, 1152, 1, 1, 1])
        agreement = np.matmul(np.transpose(caps2_predicted, (0, 1, 2, 4, 3)), caps2_output_tiled)
        raw_weights = np.add(raw_weights, agreement)


# Estimate class
y_proba = safe_norm(caps2_output, axis=-2)

digit_caps_image = y_proba.reshape(10, 1)
image.imsave(os.path.join(PATH_TO_DIGIT_CAPS, '{}.png'.format(0)), digit_caps_image, cmap=CMAP, vmin=0, vmax=1)


################################################################################
# Prediction
################################################################################

y_proba_argmax = np.argmax(y_proba, axis=2)
y_pred = np.squeeze(y_proba_argmax, axis=[1,2])

print('Prediction: {}'.format(y_pred))


################################################################################
# Reconstruction
################################################################################

caps2_output = np.squeeze(caps2_output)
reconstruction_input = caps2_output[y_pred]

output = reconstruction_input

json_params = { 'vector': output.tolist(), 'prediction': int(y_pred)}
with open(os.path.join(PATH_TO_RECONSTRUCTION_JSON_PARAMS, '{}.json'.format(0)), 'w') as outfile:
    json.dump(json_params, outfile)

fully_connected1 = fully_connected1.reshape(10, 16, 512)[y_pred]

signal = np.dot(output, fully_connected1) + fully_connected1_bias # bias
output = ReLU_function(signal)

signal = np.dot(output, fully_connected2) + fully_connected2_bias # bias
output = ReLU_function(signal)

signal = np.dot(output, fully_connected3) + fully_connected3_bias # bias
output = sigmoid_function(signal)

output = output.reshape(28,28)

image.imsave(os.path.join(PATH_TO_RECONSTRUCTION, '{}.png'.format(0)), output, cmap=CMAP, vmin=0, vmax=1)
