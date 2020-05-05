import os
import fnmatch
import re
import numpy as np
from PIL import Image
import uuid
import matplotlib.image as image
from matplotlib.colors import LinearSegmentedColormap
from flask import Flask, send_from_directory, jsonify, request

app = Flask(__name__, static_folder='client/build')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path == '':
        return send_from_directory('client/build', 'index.html')
    else:
        if os.path.exists('client/build/' + path):
            return send_from_directory('client/build', path)
        if os.path.exists(path):
            return send_from_directory('', path)
        else:
            return send_from_directory('client/build', 'index.html')


tokenize = re.compile(r'(\d+)|(\D+)').findall
def natural_sortkey(string):
    return tuple(int(num) if num else alpha for num, alpha in tokenize(string))


@app.route('/images')
def get_images():
    res = {}
    for input_dir in os.listdir('visualizations'):
        res[input_dir] = []
        layers_path = os.path.join('visualizations', input_dir)
        if os.path.isdir(layers_path):
            for i, layer_dir in enumerate(sorted(os.listdir(layers_path), key=natural_sortkey)):
                res[input_dir].append([])
                images_path = os.path.join(layers_path, layer_dir)
                if os.path.isdir(images_path):
                    for image_file in sorted(os.listdir(images_path), key=natural_sortkey):
                        if fnmatch.fnmatch(image_file, '*.png') or fnmatch.fnmatch(image_file, '*.json'):
                            image_path = os.path.join(images_path, image_file)
                            res[input_dir][i].append(image_path)
    return jsonify(res)


expit = lambda x: 1.0 / (1 + np.exp(-x))
def sigmoid_function(signal):
    # Prevent overflow.
    signal = np.clip(signal, -500, 500)

    # Calculate activation signal
    return expit(signal)


def ReLU_function(signal):
    # Return the activation signal
    return np.maximum(0, signal)


@app.route('/api/reconstruct', methods=['POST'])
def reconstruct():
    CMAP = LinearSegmentedColormap.from_list('greyscale', ((0, 0, 0), (1, 1, 1)), N=256, gamma=1.0)

    PATH_TO_WEIGHTS = 'numpy_weights'
    PATH_TO_WEIGHTS_FULLY_CONNECTED1 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected1.weights.npz')
    PATH_TO_WEIGHTS_FULLY_CONNECTED2 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected2.weights.npz')
    PATH_TO_WEIGHTS_FULLY_CONNECTED3 = os.path.join(PATH_TO_WEIGHTS, 'fully_connected3.weights.npz')
    PATH_TO_WEIGHTS_FULLY_CONNECTED1_BIAS = os.path.join(PATH_TO_WEIGHTS, 'fully_connected1.bias.npz')
    PATH_TO_WEIGHTS_FULLY_CONNECTED2_BIAS = os.path.join(PATH_TO_WEIGHTS, 'fully_connected2.bias.npz')
    PATH_TO_WEIGHTS_FULLY_CONNECTED3_BIAS = os.path.join(PATH_TO_WEIGHTS, 'fully_connected3.bias.npz')

    fully_connected1 = np.load(PATH_TO_WEIGHTS_FULLY_CONNECTED1)
    fully_connected2 = np.load(PATH_TO_WEIGHTS_FULLY_CONNECTED2)
    fully_connected3 = np.load(PATH_TO_WEIGHTS_FULLY_CONNECTED3)
    fully_connected1_bias = np.load(PATH_TO_WEIGHTS_FULLY_CONNECTED1_BIAS)
    fully_connected2_bias = np.load(PATH_TO_WEIGHTS_FULLY_CONNECTED2_BIAS)
    fully_connected3_bias = np.load(PATH_TO_WEIGHTS_FULLY_CONNECTED3_BIAS)

    output = np.array(request.json['vector'])

    fully_connected1 = fully_connected1.reshape(10, 16, 512)[request.json['predicted']]

    signal = np.dot(output, fully_connected1) + fully_connected1_bias # bias
    output = ReLU_function(signal)

    signal = np.dot(output, fully_connected2) + fully_connected2_bias # bias
    output = ReLU_function(signal)

    signal = np.dot(output, fully_connected3) + fully_connected3_bias # bias
    output = sigmoid_function(signal)

    output = output.reshape(28,28)

    folder = 'reconstructions'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    path = os.path.join('reconstructions', '{}.png').format(str(uuid.uuid4()))
    image.imsave(path, output, cmap=CMAP, vmin=0, vmax=1)

    return jsonify({'url': path})


if __name__ == '__main__':
    app.run(debug=True)
