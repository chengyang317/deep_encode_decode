import tensorflow as tf
import prettytensor as pt
import numpy as np


class NetWork(object):
    """
    Net work for the encode_decode architechture.
    """
    def __init__(self, batch_size):
        input_tensor = tf.placeholder(tf.float32, shape=(batch_size, DATA_SIZE))
        label_tensor = tf.placeholder(tf.float32, shape=(BATCH_SIZE, CLASSES))
        pretty_input = pt.wrap(input_tensor)
