from PIL import Image
import numpy as np
import tensorflow as tf
import os
import scipy.misc
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

n_classes = 24

# def prepare_label(input_batch, new_size, one_hot=True):
#     """Resize masks and perform one-hot encoding.

#     Args:
#       input_batch: input tensor of shape [batch_size H W 1].
#       new_size: a tensor with new height and width.

#     Returns:
#       Outputs a tensor of shape [batch_size h w 21]
#       with last dimension comprised of 0's and 1's only.
#     """
#     with tf.name_scope('label_encode'):
#       input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
#       input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
#       if one_hot:
#         input_batch = tf.one_hot(input_batch, depth=n_classes)
#     return input_batch

def prepare_dense_label(input_batch, new_size, one_hot=True):
  """Resize masks and perform one-hot encoding.

  Args:
    input_batch: input tensor of shape [batch_size H W 1].
    new_size: a tensor with new height and width.

  Returns:
    Outputs a tensor of shape [batch_size h w 21]
    with last dimension comprised of 0's and 1's only.
  """
  with tf.name_scope('label_encode'):
    print(input_batch)
    input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interpolation
    # input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
    if one_hot:
      input_batch = tf.one_hot(input_batch, depth=n_classes)
  return input_batch