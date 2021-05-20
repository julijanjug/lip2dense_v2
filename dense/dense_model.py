import tensorflow as tf
from utils.ops import *

def dense_net(image, name):
  with tf.variable_scope(name) as scope:
      is_BN = False
      print("---------")
      print(image)
      dense_conv1 = conv2d(image, 512, 3, 1, relu=True, bn=is_BN, name='dense_conv1')
      dense_conv2 = conv2d(dense_conv1, 512, 3, 1, relu=True, bn=is_BN, name='dense_conv2')
      dense_conv3 = conv2d(dense_conv2, 512, 3, 1, relu=True, bn=is_BN, name='dense_conv3')
      dense_conv4 = conv2d(dense_conv3, 512, 3, 1, relu=True, bn=is_BN, name='dense_conv4')
      dense_conv5 = conv2d(dense_conv4, 512, 3, 1, relu=True, bn=is_BN, name='dense_conv5')
      dense_conv6 = conv2d(dense_conv5, 512, 3, 1, relu=True, bn=is_BN, name='dense_conv6')

      dense_conv7 = conv2d(dense_conv6, 512, 1, 1, relu=True, bn=is_BN, name='dense_conv7')
      dense_conv8 = conv2d(dense_conv7, 24, 1, 1, relu=False, bn=is_BN, name='dense_conv8') #popravil sem iz 512 v 24 zaradi st classov

      return dense_conv8, dense_conv6