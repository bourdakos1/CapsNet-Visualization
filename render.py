import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as image
from scipy import signal
from matplotlib.colors import LinearSegmentedColormap

CMAP = LinearSegmentedColormap.from_list('greyscale', ((0, 0, 0), (1, 1, 1)), N=256, gamma=1.0)

PATH_TO_VISUALIZATIONS = 'visualizations'
PATH_TO_CONV1_KERNEL = os.path.join(PATH_TO_VISUALIZATIONS, '0_conv1_kernel')
PATH_TO_CONV1 = os.path.join(PATH_TO_VISUALIZATIONS, '1_conv1')
PATH_TO_RELU = os.path.join(PATH_TO_VISUALIZATIONS, '2_conv1+relu')
PATH_TO_PRIMARY_CAPS = os.path.join(PATH_TO_VISUALIZATIONS, '3_primary_caps')

PATH_TO_WEIGHTS = 'numpy_weights'
PATH_TO_WEIGHTS_CONV1 = os.path.join(PATH_TO_WEIGHTS, 'conv1.weights.npz')
PATH_TO_WEIGHTS_PRIMARY_CAPS = os.path.join(PATH_TO_WEIGHTS, 'primary_caps.weights.npz')
PATH_TO_WEIGHTS_DIGIT_CAPS = os.path.join(PATH_TO_WEIGHTS, 'digit_caps.weights.npz')
PATH_TO_WEIGHTS_FULLY_CONNECTED1 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected1.weights.npz')
PATH_TO_WEIGHTS_FULLY_CONNECTED2 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected2.weights.npz')
PATH_TO_WEIGHTS_FULLY_CONNECTED3 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected3.weights.npz')

conv1 = np.load(PATH_TO_WEIGHTS_CONV1)
primary_caps = np.load(PATH_TO_WEIGHTS_PRIMARY_CAPS)
digit_caps = np.load(PATH_TO_WEIGHTS_DIGIT_CAPS)

output = np.empty((20, 20, 256))
primary_caps_output = np.empty((6, 6, 256))


def squash(s, axis=-1, epsilon=1e-7):
    squared_norm = np.sum(np.square(s), axis=axis, keepdims=True)
    safe_norm = np.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = s / safe_norm
    return squash_factor * unit_vector

def safe_norm(s, axis=-1, epsilon=1e-7, keepdims=False):
    squared_norm = np.sum(np.square(s), axis=axis, keepdims=keepdims)
    return np.sqrt(squared_norm + epsilon)


for i in range(conv1.shape[3]):
    # Get the 9x9x1 filter:
    extracted_filter = conv1[:, :, :, i]

    # Get rid of the last dimension (hence get 9x9):
    extracted_filter = np.squeeze(extracted_filter)

    image.imsave(os.path.join(PATH_TO_CONV1_KERNEL, '{}.png'.format(i)), extracted_filter)

    img = Image.open('0.png')
    arr2 = np.array(img.getdata(), dtype=np.uint8)

    arr2 = np.resize(arr2, (img.size[1], img.size[0], 4))
    arr2 = arr2[:,:,1]

    conv = signal.correlate2d(arr2, extracted_filter, 'valid')

    # I ignore vmin because then it will premeturely ReLU.
    image.imsave(os.path.join(PATH_TO_CONV1, '{}.png'.format(i)), conv, cmap=CMAP, vmax=255)

    relu = np.maximum(conv, 0, conv)
    output[:, :, i] = relu

    image.imsave(os.path.join(PATH_TO_RELU, '{}.png'.format(i)), relu, cmap=CMAP, vmin=0, vmax=255)


for i in range(primary_caps.shape[3]):
    extracted_filter = primary_caps[:, :, :, i]
    conv2 = signal.correlate(output, extracted_filter, 'valid')
    # Outputs 12x12, but we need 6x6 so we need to somehow drop every other item
    conv2 = conv2[::2, ::2, :]

    conv2 = np.squeeze(conv2)
    primary_caps_output[:, :, i] = conv2


# then we reshape to 6x6x8x32
# primary_caps_output = np.reshape(primary_caps_output, (6,6,8,32))
# We actually just want to flatten it to a 1152x8 matrix?
primary_caps_output = np.reshape(primary_caps_output, (-1, 6 * 6 * 32, 8))
squashed_caps_output = squash(primary_caps_output)


# x = np.apply_along_axis(np.linalg.norm, 2, squashed_caps_output)
#
# for layer in range(32):
#     image.imsave(os.path.join(PATH_TO_PRIMARY_CAPS, '{}.png'.format(layer)), x[:,:,layer])


digit_caps = np.swapaxes(digit_caps, 3, 4)

caps_output_expanded = np.expand_dims(squashed_caps_output, -1)
caps_output_tile = np.expand_dims(caps_output_expanded, 2)
caps_output_tiled = np.tile(caps_output_tile, [1, 1, 10, 1, 1])

caps2_predicted = np.matmul(digit_caps, caps_output_tiled)

# Round 1
raw_weights = np.zeros([1, 1152, 10, 1, 1])

# Softmax
routing_weights = np.exp(raw_weights) / np.sum(np.exp(raw_weights), axis=2, keepdims=True)
# Sumation of the element wise multiplication
weighted_predictions = np.multiply(routing_weights, caps2_predicted)
weighted_sum = np.sum(weighted_predictions, axis=1, keepdims=True)
# Squash
caps2_output_round_1 = squash(weighted_sum, axis=-2)

# Add dot product to the weights
caps2_output_round_1_tiled = np.tile(caps2_output_round_1, [1, 1152, 1, 1, 1])
agreement = np.matmul(np.transpose(caps2_predicted, (0, 1, 2, 4, 3)), caps2_output_round_1_tiled)
raw_weights_round_2 = np.add(raw_weights, agreement)

# Round 2

# Softmax
routing_weights_round_2 = np.exp(raw_weights_round_2) / np.sum(np.exp(raw_weights_round_2), axis=2, keepdims=True)
# Sumation of the element wise multiplication
weighted_predictions_round_2 = np.multiply(routing_weights_round_2, caps2_predicted)
weighted_sum_round_2 = np.sum(weighted_predictions_round_2, axis=1, keepdims=True)
# Squash
caps2_output_round_2 = squash(weighted_sum_round_2, axis=-2)

caps2_output = caps2_output_round_2

# Estimate class
y_proba = safe_norm(caps2_output, axis=-2)

y_proba_argmax = np.argmax(y_proba, axis=2)
y_pred = np.squeeze(y_proba_argmax, axis=[1,2])

print(y_pred)
