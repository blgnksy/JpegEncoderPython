import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy.fftpack import dct
from tqdm import tqdm
from itertools import *
import pandas as pd

import quantization_matrix as qm

zz_order_list = [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0), (3, 1), (2, 2),
                 (1, 3), (0, 4), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2), (3, 3), (2, 4),
                 (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3),
                 (4, 4), (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (7, 3), (6, 4), (5, 5),
                 (4, 6), (3, 7), (4, 7), (5, 6), (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)]

previous_dc = 0.0

huff_t1_df = pd.read_csv('./huff_t1.csv')


def rgb2y(filename):
    _img_rgb = misc.imread(filename)
    _width, _height, _ = _img_rgb.shape
    _img_y = np.empty((_width, _height), dtype=np.float)
    _img_y[:, :] = (0.299 * _img_rgb[:, :, 0] + 0.587 * _img_rgb[:, :, 1] + 0.114 * _img_rgb[:, :,
                                                                                    2]) - 128.0  # Zero shifting
    return _img_y, _width, _height


def image2chunks2dct(image):
    _dct_chunks = []
    _w, _h = image.shape
    l = _w * _h * 8 * 8
    for i in tqdm(range(0, _w / 8)):
        for j in range(0, _h / 8):
            _dct_chunks.append(dct(image[i * 8: i * 8 + 8, j * 8: j * 8 + 8], 2))
    return _dct_chunks


def quantize(dct_chunks):
    _quant_chunks = []
    for dct_chunk in tqdm(dct_chunks):
        temp = np.round(dct_chunk / qm.q_IrfanV_040)
        _quant_chunks.append(temp)
    return _quant_chunks


def order_chunks(q_chunks):
    _ordered_chunk = []
    for i in tqdm(range(len(zz_order_list))):
        _ordered_chunk.append(q_chunks[0][zz_order_list[i]])  # for just one chunk
    return _ordered_chunk


# I used the same code from https://www.techrepublic.com/article/run-length-encoding-in-python
# for this function
def run_length_code(ordered_chunk):
    rlc = [(len(list(group)), name) for name, group in groupby(ordered_chunk)]
    return rlc


def hufmann(run_length_coded):
    print(run_length_coded[0])
    return 0


def huffmann_T1(amplitude):
    size = int(math.log(abs(amplitude), 2)) + 1
    return huff_t1_df['Code'][size]


def huffmann_T2(amplitude):
    size = int(math.log(abs(amplitude), 2)) + 1
    code = None
    if size == 0:
        code = None
    else:
        if amplitude > 0:
            to_code = pow(2, size - 1)
            code = bin((amplitude - to_code) | (1 << (size - 1)))
        else:
            to_code = pow(2, size) - 1
            code = binbits((to_code + amplitude), size)
    return code


def huffmann_T3():
    return 0


def binbits(x, n):
    """Return binary representation of x with at least n bits"""
    bits = bin(x).split('b')[1]

    if len(bits) < n:
        return '0b' + '0' * (n - len(bits)) + bits


if __name__ == '__main__':
    # Image Reading From File and RGB Converting to Luminance Channel.
    img_y, w, h = rgb2y(filename='./chess.bmp')
    print('---------RGB Converting to Luminance Channel. Done!--------')

    # Image Dividing Into Chunks, and DCT starting.
    dct_chunks = image2chunks2dct(image=img_y)
    print('---------Image Divided Into Chunks, and DCT. Done!---------')

    # Quantization Starting.
    q_chunks = quantize(dct_chunks=dct_chunks)
    print('------------------Quantization Finished.-------------------')

    # Zig-zag Ordering the DCT Coefficients
    ordered_chunk = order_chunks(q_chunks=q_chunks)
    print('------------------Chunks Zig-Zag Ordered.-------------------')

    run_length_coded = run_length_code(ordered_chunk=ordered_chunk)

    hufmann(run_length_coded=run_length_coded)

    plt.interactive(True)
    plt.imshow(img_y, cmap='gray')
