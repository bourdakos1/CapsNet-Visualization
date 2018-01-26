import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as image
from scipy import signal
from matplotlib.colors import LinearSegmentedColormap

PATH_TO_TEST_IMAGES_DIR = 'test_images'
PATH_TO_TEST_IMAGE = '9_2.png'

# Black and white color map going from (0, 0, 0) "black" to (1, 1, 1) "white".
CMAP = LinearSegmentedColormap.from_list('greyscale', ((0, 0, 0), (1, 1, 1)), N=256, gamma=1.0)

# Output directories for visualizations.
PATH_TO_ROOT = 'visualizations'
PATH_TO_VISUALIZATIONS = os.path.join(PATH_TO_ROOT, PATH_TO_TEST_IMAGE)
if not os.path.exists(PATH_TO_VISUALIZATIONS):
    os.mkdir(PATH_TO_VISUALIZATIONS)

PATH_TO_CONV1_KERNEL = os.path.join(PATH_TO_VISUALIZATIONS, '0')
PATH_TO_CONV1 = os.path.join(PATH_TO_VISUALIZATIONS, '1')
PATH_TO_RELU = os.path.join(PATH_TO_VISUALIZATIONS, '2')
PATH_TO_PRIMARY_CAPS = os.path.join(PATH_TO_VISUALIZATIONS, '3')

if not os.path.exists(PATH_TO_CONV1_KERNEL):
    os.mkdir(PATH_TO_CONV1_KERNEL)
if not os.path.exists(PATH_TO_CONV1):
    os.mkdir(PATH_TO_CONV1)
if not os.path.exists(PATH_TO_RELU):
    os.mkdir(PATH_TO_RELU)
if not os.path.exists(PATH_TO_PRIMARY_CAPS):
    os.mkdir(PATH_TO_PRIMARY_CAPS)

# Input directories for layer weights.
PATH_TO_WEIGHTS = 'numpy_weights'
PATH_TO_WEIGHTS_CONV1 = os.path.join(PATH_TO_WEIGHTS, 'conv1.weights.npz')
PATH_TO_WEIGHTS_CONV1_BIAS = os.path.join(PATH_TO_WEIGHTS, 'conv1.bias.npz')
PATH_TO_WEIGHTS_PRIMARY_CAPS = os.path.join(PATH_TO_WEIGHTS, 'primary_caps.weights.npz')
PATH_TO_WEIGHTS_PRIMARY_CAPS_BIAS = os.path.join(PATH_TO_WEIGHTS, 'primary_caps.bias.npz')
PATH_TO_WEIGHTS_DIGIT_CAPS = os.path.join(PATH_TO_WEIGHTS, 'digit_caps.weights.npz')

# Number of routing iterations to do in DigitCaps layer.
NUMBER_OF_ROUNDS = 3

# Load the weights with numpy.
conv1 = np.load(PATH_TO_WEIGHTS_CONV1)
conv1_bias = np.load(PATH_TO_WEIGHTS_CONV1_BIAS)
primary_caps = np.load(PATH_TO_WEIGHTS_PRIMARY_CAPS)
primary_caps_bias = np.load(PATH_TO_WEIGHTS_PRIMARY_CAPS_BIAS)
digit_caps = np.load(PATH_TO_WEIGHTS_DIGIT_CAPS)

output = np.empty((20, 20, 256))
primary_caps_output = np.empty((6, 6, 256))


def squash(s, axis=-1, epsilon=1e-9):
    squared_norm = np.sum(np.square(s), axis=axis, keepdims=True)
    safe_norm = np.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = s / safe_norm
    return squash_factor * unit_vector


def safe_norm(s, axis=-1, epsilon=1e-9, keepdims=False):
    squared_norm = np.sum(np.square(s), axis=axis, keepdims=keepdims)
    return np.sqrt(squared_norm + epsilon)


for i in range(conv1.shape[3]):
    # Get the 9x9x1 filter:
    extracted_filter = conv1[:, :, :, i]

    # Get rid of the last dimension (hence get 9x9):
    extracted_filter = np.squeeze(extracted_filter)

    image.imsave(os.path.join(PATH_TO_CONV1_KERNEL, '{}.png'.format(i)), extracted_filter)

    img = Image.open(os.path.join(PATH_TO_TEST_IMAGES_DIR, PATH_TO_TEST_IMAGE))
    arr2 = np.array(img.getdata(), dtype=np.uint8)

    arr2 = np.resize(arr2, (img.size[1], img.size[0], 4))
    arr2 = arr2[:, :, 1]

    conv = signal.correlate2d(arr2, extracted_filter, 'valid')

    # I ignore vmin because then it will premeturely ReLU.
    image.imsave(os.path.join(PATH_TO_CONV1, '{}.png'.format(i)), conv, cmap=CMAP, vmax=255)

    conv = conv + conv1_bias[i]

    conv = np.maximum(0, conv)
    output[:, :, i] = conv

    image.imsave(os.path.join(PATH_TO_RELU, '{}.png'.format(i)), conv, cmap=CMAP, vmin=0, vmax=255)


for i in range(primary_caps.shape[3]):
    # 9,9,256,256
    extracted_filter = primary_caps[:, :, :, i]

    conv2 = signal.correlate(output, extracted_filter, 'valid')

    # Outputs 12x12, but we need 6x6 so we need to somehow drop every other item
    conv2 = conv2[::2, ::2, :]

    conv2 = np.squeeze(conv2)

    conv2 = conv2 + primary_caps_bias[i]

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


# This is completely useless, but it makes it easier to visualize
normed_squashed_caps_output = np.reshape(safe_norm(squashed_caps_output, axis=-2), (6, 6, 32))
for i in range(normed_squashed_caps_output.shape[2]):
    image.imsave(os.path.join(PATH_TO_PRIMARY_CAPS, '{}.png'.format(i)), normed_squashed_caps_output[:, :, i], cmap=CMAP, vmin=0, vmax=1)


# Add in a blank dimension: 1152, 1, 8, 1
squashed_caps_output = np.reshape(squashed_caps_output, (-1, 1, squashed_caps_output.shape[-2], 1))
# Tile the capsules in the inserted dimension: 1152, 10, 8, 1
caps_output_tiled = np.tile(squashed_caps_output, [1, 1, 10, 1, 1])

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
    if x != NUMBER_OF_ROUNDS - 1:
        # Add dot product to the weights
        caps2_output_tiled = np.tile(caps2_output, [1, 1152, 1, 1, 1])
        agreement = np.matmul(np.transpose(caps2_predicted, (0, 1, 2, 4, 3)), caps2_output_tiled)
        raw_weights = np.add(raw_weights, agreement)


# Estimate class
y_proba = safe_norm(caps2_output, axis=-2)
print(y_proba)

y_proba_argmax = np.argmax(y_proba, axis=2)
y_pred = np.squeeze(y_proba_argmax, axis=[1,2])

print(y_pred)
