import os
import fnmatch
import re
from flask import Flask, send_from_directory, jsonify

app = Flask(__name__, static_folder='')


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
        res[input_dir] = {}
        layers_path = os.path.join('visualizations', input_dir)
        if os.path.isdir(layers_path):
            for layer_dir in os.listdir(layers_path):
                res[input_dir][layer_dir] = []
                images_path = os.path.join(layers_path, layer_dir)
                if os.path.isdir(images_path):
                    for image_file in sorted(os.listdir(images_path), key=natural_sortkey):
                        if fnmatch.fnmatch(image_file, '*.png'):
                            image_path = os.path.join(images_path, image_file)
                            res[input_dir][layer_dir].append(image_path)
    return jsonify(res)


if __name__ == '__main__':
    app.run(debug=True)
