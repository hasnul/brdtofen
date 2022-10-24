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
import tqdm
from termcolor import colored

from constants import (
    TILES_DIR, NN_MODEL_PATH, FEN_CHARS, USE_GRAYSCALE, DETECT_CORNERS
)
from utils import compressed_fen
from train import image_data
from chessboard_finder import get_chessboard_corners
from chessboard_image import _get_resized_chessboard

OUT_FILE = "debug.html"
MEM_LIMIT = 32 * 1024 * 1024  # 32 MBytes; this is NOT a "limit" on mem used by prog
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

    return np.array([img_data[r:r+SQSIZE, c:c+SQSIZE]
                     for r in range(0, BRDSIZE, SQSIZE)
                     for c in range(0, BRDSIZE, SQSIZE)])


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

    total_size = 0
    for path in img_paths:
        total_size += os.path.getsize(path)

    # TODO: What if total size < MEM_LIMIT or is not that 'big'?
    # call do_the_whole_batch()
    # return

    # Assuming a more or less uniform image file size
    batches = total_size//MEM_LIMIT + 1 if total_size % MEM_LIMIT != 0 else total_size//MEM_LIMIT
    batch_size = len(img_paths)//batches + 1 if len(img_paths) % batches != 0 else len(img_paths)//batches

    CHUNKSIZE = 2
    batch_result = []
    with Pool() as pool:
        if not quiet:
            print("Loading model ...")
        model = models.load_model(NN_MODEL_PATH)
        for batch in range(batches):
            if batch != batches - 1:
                batch_paths = img_paths[batch*batch_size: (batch+1)*batch_size]
            else:
                batch_paths = img_paths[batch*batch_size:]
            imap_iter = pool.imap(_chessboard_tiles_img_data, batch_paths, CHUNKSIZE)
            if not quiet:
                print(f"Processing batch {batch+1}/{batches} ...")
                tqdm_iter = tqdm.tqdm(imap_iter, total=len(batch_paths))
                squares_iter = tqdm_iter
            else:
                squares_iter = imap_iter

            #img_data_list = []
            if batch != batches - 1:
                img_data = np.zeros((batch_size*64, SQSIZE, SQSIZE, 1))
            else:
                if len(img_paths) % batches != 0:
                    last_batch = len(img_paths) - (batches - 1)*batch_size
                else:
                    last_batch = batch_size
                img_data = np.zeros((last_batch*64, SQSIZE, SQSIZE, 1))
            for i, sq in enumerate(squares_iter):
                #img_data_list.extend(sq)
                img_data[i*64:(i+1)*64, :, :, 0] = np.moveaxis(sq, -1, 0)

            if not quiet:
                print(f"Running prediction for batch {batch+1}/{batches} ...")
                verbose = 1
            else:
                verbose = 0

            result = model.predict(img_data, batch_size=32, verbose=verbose)
            #result = model(np.array(img_data_list))

            if not quiet:
                print("Processing results ...")
            result = np.reshape(result, (len(batch_paths), 64, len(FEN_CHARS)))

            max_probabilities = np.max(result, axis=2)
            confidence = np.prod(max_probabilities, axis=1)
            fen_indices = np.argmax(result, axis=2)

            fens = [makefen_from_indices(indices) for indices in fen_indices]

            batch_result.extend([{"file": os.path.basename(batch_paths[i]), "confidence": confidence[i],
                                "tile_prob": max_probabilities[i], "fen": fens[i]}
                                for i in range(len(batch_paths))])

    return batch_result


def makefen_from_indices(indices):
    raw_fen = "".join(list(map(lambda c: FEN_CHARS[c], indices)))
    split_fen = "/".join(raw_fen[i:i+8] for i in range(0, len(raw_fen), 8))
    return compressed_fen(split_fen)


def report(r, color=False):
    confidence = r['confidence']
    fen = r['fen']
    file = r['file']
    quality = validate_fen(fen)
    if color:
        c = colored(quality, 'green') if quality == "OK" else colored(quality, 'red')
    else:
        c = quality
    print(f'{{"file": "{file}", "confidence": {confidence:0.08f}, "fen": "{fen}", "status": "{c}"}}')
    img = os.path.join('data', r['file'])
    _save_output_html(img, fen, [t for t in r['tile_prob']], confidence)

    return quality

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
    parser.add_argument("-c", "--color", help="Color the status text",
                        action="store_true")
    parser.add_argument("-d", "--debug", help="Saves debug output to debug.html",
                        action="store_true")
    parser.add_argument("image_path", help="Path/glob to PNG chessboard image(s)")
    args = parser.parse_args()

    if not args.quiet:
        print('Tensorflow {}'.format(tf.version.VERSION))

    if len(sys.argv) > 1:

        import time
        start = time.time()
        with open(OUT_FILE, "w") as f:
            f.write('<link rel="stylesheet" href="./web/style.css" />')

        image_path = os.path.expanduser(args.image_path)
        paths = [path for path in sorted(glob(image_path))]
        result = predict_chessboard(paths, args.quiet)
        bad = 0
        total = 0
        for r in result:
            if report(r, color=args.color) == "BAD":
                bad += 1
            total += 1

        good = (total - bad)/total
        print()
        print(f"OK = {total - bad}/{total} === {good*100:.2f}% (optimistic)")

        end = time.time()
        print(f"Time taken = {end - start:.3f} seconds")
        print(f"Prediction rate = {total/(end - start):.0f} images per second")
