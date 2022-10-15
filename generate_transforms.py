#!/usr/bin/env python3

import os
from glob import glob
import random
import cv2
import numpy as np

from constants import CHESSBOARDS_DIR, TRANSFORM_DIR, USE_GRAYSCALE

OVERWRITE = False
MIN_ROT = 1  # minimum rotation magnitude in degrees
MAX_ROT = 3


def rotate(img):
    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    scale = 1.0
    angle = random.choice([-1, 1]) * random.uniform(MIN_ROT, MAX_ROT)
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    return cv2.warpAffine(img, M, (w,h))


def transform(img, method):
    if (method == "rotate"):
        return rotate(img)

    # Might consider other transforms such as
    # - shear
    # - translation
    # - fractional scaling (effectively creating a border on one or more sides)
    # - convex/concave deformations or warping
    # - pincushion/barrel distortion

    raise f"Unknown transform method: {method}"


if __name__ == '__main__':

    if not os.path.exists(TRANSFORM_DIR):
        os.makedirs(TRANSFORM_DIR)

    chessboard_paths = glob("{}/*/*.png".format(CHESSBOARDS_DIR))
    num_chessboards = len(chessboard_paths)
    success = 0
    skipped = 0
    failed = 0
    method = "rotate"

    for i, path in enumerate(chessboard_paths):

        print("%3d/%d %s" % (i + 1, num_chessboards, path))

        path_split = path.split('/')
        chessboard_dir = path_split[-2]
        filename = path_split[-1]

        save_folder = os.path.join(TRANSFORM_DIR, method + "_" + chessboard_dir)
        full_save_path = os.path.join(save_folder, filename) 

        if os.path.exists(full_save_path) and not OVERWRITE:
            print("\tIgnoring existing {}\n".format(full_save_path))
            skipped += 1
            continue
        else:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

        img = cv2.imread(path)
        tx = transform(img, method)
        if cv2.imwrite(os.path.abspath(full_save_path), tx):
            success += 1
        else:
            failed += 1

    print('Processed {} chessboard images ({} generated, {} skipped, {} write failed)'
          .format(num_chessboards, success, skipped, failed))
