import numpy as np
import PIL.Image
BRDSIZE = 256
SQSIZE = BRDSIZE // 8
assert (BRDSIZE % 8 == 0)
NUMCHAN = 3


def _get_resized_chessboard(chessboard_img_path):
    """ chessboard_img_path = path to a chessboard image
        Returns a 256x256 image of a chessboard (32x32 per tile)
    """
    img_data = PIL.Image.open(chessboard_img_path).convert('RGB')
    return img_data.resize([BRDSIZE, BRDSIZE], PIL.Image.BILINEAR)


def get_chessboard_tiles(chessboard_img_path, use_grayscale=True):
    """ chessboard_img_path = path to a chessboard image
        use_grayscale = true/false for whether to return tiles in grayscale

        Returns a list (length 64) of 32x32 image data
    """

    img_data = _get_resized_chessboard(chessboard_img_path)

    if use_grayscale:
        img_data = img_data.convert('L', (0.2989, 0.5870, 0.1140, 0))
        img_data = np.repeat(img_data, NUMCHAN)
        img_data = np.reshape(img_data, (BRDSIZE, BRDSIZE, NUMCHAN))

    board = np.asarray(img_data, dtype=np.uint8)
    squares = [board[r : r + SQSIZE, c : c + SQSIZE]
               for r in range(0, BRDSIZE, SQSIZE)
               for c in range(0, BRDSIZE, SQSIZE)]

    return list(map(lambda x : PIL.Image.fromarray(x, 'RGB'), squares))
