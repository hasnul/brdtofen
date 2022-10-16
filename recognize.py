#!/usr/bin/env python3

import sys
from glob import glob
from io import BytesIO
from multiprocessing import Pool
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from termcolor import colored

from constants import (
    TILES_DIR, NN_MODEL_PATH, FEN_CHARS, USE_GRAYSCALE, DETECT_CORNERS
)
from utils import compressed_fen
from train import image_data
from chessboard_finder import get_chessboard_corners
from chessboard_image import _get_resized_chessboard

OUT_FILE = "debug.html"

BRDSIZE = 256
SQSIZE = BRDSIZE // 8
assert (BRDSIZE % 8 == 0)
NUMCHAN = 3


def _chessboard_tiles_img_data(chessboard_img_path):
    """ Given a file path to a chessboard image, returns a
        size-64 list of 32x32xchan tensors representing each square.
    """
    board = tf.io.read_file(chessboard_img_path)
    img_data = tf.image.decode_image(board, channels=NUMCHAN)  # what happens if PNG?
    img_data = tf.image.resize(img_data, [BRDSIZE, BRDSIZE])

    if USE_GRAYSCALE:
        # The RGB weights are the same as the ones used in
        # get_chessboard_tiles(). See source code of the rgb_to_grayscale method.
        img_data = tf.image.rgb_to_grayscale(img_data)

    return [img_data[r:r+SQSIZE, c:c+SQSIZE]
                     for r in range(0, BRDSIZE, SQSIZE)
                     for c in range(0, BRDSIZE, SQSIZE)]


def _confidence_color(confidence):
    if confidence >= 0.999:
        return "#00C176"
    elif confidence > 0.99:
        return "#88C100"
    elif confidence > 0.95:
        return "#FABE28"
    elif confidence > 0.9:
        return "#FF8A00"
    else:
        return "#FF003C"


def _save_output_html(chessboard_img_path, fen, predictions, confidence):
    confidence_color = _confidence_color(confidence)
    html = '<h3>{}</h3>'.format(chessboard_img_path)
    html += '<div class="boards-row">'
    html += '<img src="{}" />'.format(chessboard_img_path)
    html += '<img src="http://www.fen-to-image.com/image/32/{}"/>'.format(fen)
    html += '<div class="predictions-matrix">'
    for i in range(8):
        html += '<div>'
        for j in range(8):
            c = predictions[i*8 + j]
            html += '<div class="prediction" style="color: {}">{}</div>'.format(
                _confidence_color(c),
                format(c, '.3f')
            )
        html += '</div>'
    html += '</div>'
    html += '</div>'
    html += '<br />'
    html += '<a href="https://lichess.org/editor/{}" target="_blank">{}</a>'.format(
        fen, fen
    )
    html += '<div style="color: {}">{}</div>'.format(confidence_color, confidence)
    html += '<br /><br />'
    with open(OUT_FILE, "a") as f:
        f.write(html)


def predict_chessboard(img_paths, quiet):
    
    img_data_list = []
    for i, p in enumerate(img_paths):
        if not quiet:
            print(f"Reading file {i+1}/{len(img_paths)} : {p}\r", end="")
        img_data_list.extend(_chessboard_tiles_img_data(p))

    if not quiet:
        print()
        print("Loading model ...")
    model = models.load_model(NN_MODEL_PATH)

    if not quiet:
        print("Running prediction ...")
    result = model.predict(np.array(img_data_list), batch_size=64, verbose=0)

    if not quiet:
        print("Processing results ...")
    result = np.reshape(result, (len(img_paths), 64, len(FEN_CHARS)))
    max_probabilities = np.max(result, axis=2)
    confidence = np.prod(max_probabilities, axis=1)
    fen_indices = np.argmax(result, axis=2)

    fens = [makefen_from_indices(indices) for indices in fen_indices]

    return [{"file": os.path.basename(img_paths[i]), "confidence": confidence[i],
            "tile_prob": max_probabilities[i], "fen": fens[i]}
            for i in range(len(img_paths))]


def makefen_from_indices(indices):
    raw_fen = "".join(list(map(lambda c: FEN_CHARS[c], indices)))
    split_fen = "/".join(raw_fen[i:i+8] for i in range(0, len(raw_fen), 8))
    return compressed_fen(split_fen)


def report(r):
    confidence = r['confidence']
    fen = r['fen']
    file = r['file']
    v = validate_fen(fen)
    c = colored(v, 'green') if v == "OK" else colored(v, 'red')
    print(f'{{"file": "{file}", "confidence": {confidence:0.08f}, "fen": "{fen}", "status": "{c}"}}')
    img = os.path.join('data', r['file'])
    _save_output_html(img, fen, [t for t in r['tile_prob']], confidence)


# Limits on piece counts for each side
# -- this will flag some compositions and unusual situations as bad
KINGS = 1
PAWNS = 8
KNIGHTS = 2
ROOKS = 2
BISHOPS = 2
QUEENS = 2  # yes two

def validate_fen(fen):
    limits = [PAWNS, KNIGHTS, BISHOPS, ROOKS, QUEENS, KINGS]
    fenchar = "pnbrqk"

    for i, c in enumerate(fenchar):
        if (fen.count(c) > limits[i] or fen.count(c.upper()) > limits[i]):
            return "BAD"

    if (fen.count('k') == 0 or fen.count('K') == 0):
        return "BAD"

    return "OK"


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quiet", help="Only print recognized FEN position",
                        action="store_true")
    parser.add_argument("-d", "--debug", help="Saves debug output to debug.html",
                        action="store_true")
    parser.add_argument("image_path", help="Path/glob to PNG chessboard image(s)")
    args = parser.parse_args()

    if not args.quiet:
        print('Tensorflow {}'.format(tf.version.VERSION))

    if len(sys.argv) > 1:

        with open(OUT_FILE, "w") as f:
            f.write('<link rel="stylesheet" href="./web/style.css" />')

        image_path = os.path.expanduser(args.image_path)
        paths = [path for path in sorted(glob(image_path))]
        result = predict_chessboard(paths, args.quiet)
        for r in result:
            report(r)
