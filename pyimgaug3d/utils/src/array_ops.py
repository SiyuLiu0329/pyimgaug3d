import numpy as np
import tensorflow as tf

def classical_round_tf(x):
    return tf.math.floor(x+0.5)

def to_channels(arr):
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),))
    for c in channels:
        c = int(c)
        res[:, :, :, c:c+1][arr == c] = 1

    return res
