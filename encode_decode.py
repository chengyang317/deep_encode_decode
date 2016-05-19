import tensorflow as tf
import prettytensor as pt
import numpy as np


class EncodeDecode(object):
    """
    Net work for the encode_decode architechture.
    """
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.pretty_input = pt.wrap(images)

    def build(self):
        with pt.defaults_scope(activation_fn=tf.nn.relu, batch_normalize=True, l2loss=0.0005):
            result = self.pretty_input.conv2d(3, 64, stride=1, edges='SAME', init=)
