from pyimgaug3d.augmentation.src.base_augmentation import BaseAugmentation
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np


class Flip(BaseAugmentation):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def __call__(self, imgs):
        res = []
        for img in imgs:
            img = tf.transpose(img, [3, 0, 1, 2])
            if self.axis == 0:
                img = tf.image.flip_left_right(img)
            elif self.axis == 1:
                img = tf.image.flip_up_down(img)
            elif self.axis == 2:
                img = tf.transpose(img, [0, 2, 3, 1])
                img = tf.image.flip_left_right(img)
                img = tf.transpose(img, [0, 3, 1, 2])


            img = tf.transpose(img, [1, 2, 3, 0])
            res.append(img)
        return res