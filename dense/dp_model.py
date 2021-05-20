# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from kaffe.tensorflow import Network
import tensorflow as tf
#THIS CODE IS NOT IN USE CURRENTLY !!!!!!!!!!!!!!!!


#this is standar resNet101 i copied from existing project
class DensePoseNetModel(Network):
     def setup(self, is_training, n_classes):
          '''Network definition.
          
          Args:
               is_training: whether to update the running mean and variance of the batch normalisation layer.
                         If the batch size is small, it is better to keep the running mean and variance of 
                         the-pretrained model frozen.
          '''
          (self.feed('data')
               .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
               .max_pool(3, 3, 2, 2, name='pool1')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))

          (self.feed('pool1')
               .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
               .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2b')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2c'))

          (self.feed('bn2a_branch1', 
                    'bn2a_branch2c')
               .add(name='res2a')
               .relu(name='res2a_relu')
               .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2a')
               .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2b')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn2b_branch2c'))

          (self.feed('res2a_relu', 
                    'bn2b_branch2c')
               .add(name='res2b')
               .relu(name='res2b_relu')
               .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2a')
               .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2b')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn2c_branch2c'))

          (self.feed('res2b_relu', 
                    'bn2c_branch2c')
               .add(name='res2c')
               .relu(name='res2c_relu')
               .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch1'))

          (self.feed('res2c_relu')
               .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2a')
               .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2b')
               .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch2c'))

          (self.feed('bn3a_branch1', 
                    'bn3a_branch2c')
               .add(name='res3a')
               .relu(name='res3a_relu')
               .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2a')
               .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2b')
               .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b1_branch2c'))

          (self.feed('res3a_relu', 
                    'bn3b1_branch2c')
               .add(name='res3b1')
               .relu(name='res3b1_relu')
               .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2a')
               .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2b')
               .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b2_branch2c'))

          (self.feed('res3b1_relu', 
                    'bn3b2_branch2c')
               .add(name='res3b2')
               .relu(name='res3b2_relu')
               .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2a')
               .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2b')
               .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b3_branch2c'))

          (self.feed('res3b2_relu', 
                    'bn3b3_branch2c')
               .add(name='res3b3')
               .relu(name='res3b3_relu')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch1'))

          (self.feed('res3b3_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4a_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch2c'))

          (self.feed('bn4a_branch1', 
                    'bn4a_branch2c')
               .add(name='res4a')
               .relu(name='res4a_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b1_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b1_branch2c'))

          (self.feed('res4a_relu', 
                    'bn4b1_branch2c')
               .add(name='res4b1')
               .relu(name='res4b1_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b2_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b2_branch2c'))

          (self.feed('res4b1_relu', 
                    'bn4b2_branch2c')
               .add(name='res4b2')
               .relu(name='res4b2_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b3_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b3_branch2c'))

          (self.feed('res4b2_relu', 
                    'bn4b3_branch2c')
               .add(name='res4b3')
               .relu(name='res4b3_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b4_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b4_branch2c'))

          (self.feed('res4b3_relu', 
                    'bn4b4_branch2c')
               .add(name='res4b4')
               .relu(name='res4b4_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b5_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b5_branch2c'))

          (self.feed('res4b4_relu', 
                    'bn4b5_branch2c')
               .add(name='res4b5')
               .relu(name='res4b5_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b6_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b6_branch2c'))

          (self.feed('res4b5_relu', 
                    'bn4b6_branch2c')
               .add(name='res4b6')
               .relu(name='res4b6_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b7_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b7_branch2c'))

          (self.feed('res4b6_relu', 
                    'bn4b7_branch2c')
               .add(name='res4b7')
               .relu(name='res4b7_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b8_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b8_branch2c'))

          (self.feed('res4b7_relu', 
                    'bn4b8_branch2c')
               .add(name='res4b8')
               .relu(name='res4b8_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b9_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b9_branch2c'))

          (self.feed('res4b8_relu', 
                    'bn4b9_branch2c')
               .add(name='res4b9')
               .relu(name='res4b9_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b10_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b10_branch2c'))

          (self.feed('res4b9_relu', 
                    'bn4b10_branch2c')
               .add(name='res4b10')
               .relu(name='res4b10_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b11_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b11_branch2c'))

          (self.feed('res4b10_relu', 
                    'bn4b11_branch2c')
               .add(name='res4b11')
               .relu(name='res4b11_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b12_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b12_branch2c'))

          (self.feed('res4b11_relu', 
                    'bn4b12_branch2c')
               .add(name='res4b12')
               .relu(name='res4b12_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b13_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b13_branch2c'))

          (self.feed('res4b12_relu', 
                    'bn4b13_branch2c')
               .add(name='res4b13')
               .relu(name='res4b13_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b14_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b14_branch2c'))

          (self.feed('res4b13_relu', 
                    'bn4b14_branch2c')
               .add(name='res4b14')
               .relu(name='res4b14_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b15_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b15_branch2c'))

          (self.feed('res4b14_relu', 
                    'bn4b15_branch2c')
               .add(name='res4b15')
               .relu(name='res4b15_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b16_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b16_branch2c'))

          (self.feed('res4b15_relu', 
                    'bn4b16_branch2c')
               .add(name='res4b16')
               .relu(name='res4b16_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b17_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b17_branch2c'))

          (self.feed('res4b16_relu', 
                    'bn4b17_branch2c')
               .add(name='res4b17')
               .relu(name='res4b17_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b18_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b18_branch2c'))

          (self.feed('res4b17_relu', 
                    'bn4b18_branch2c')
               .add(name='res4b18')
               .relu(name='res4b18_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b19_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b19_branch2c'))

          (self.feed('res4b18_relu', 
                    'bn4b19_branch2c')
               .add(name='res4b19')
               .relu(name='res4b19_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b20_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b20_branch2c'))

          (self.feed('res4b19_relu', 
                    'bn4b20_branch2c')
               .add(name='res4b20')
               .relu(name='res4b20_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b21_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b21_branch2c'))

          (self.feed('res4b20_relu', 
                    'bn4b21_branch2c')
               .add(name='res4b21')
               .relu(name='res4b21_relu')
               .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2a')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2a')
               .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b22_branch2b')
               .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2b')
               .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c')
               .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b22_branch2c'))

          (self.feed('res4b21_relu', 
                    'bn4b22_branch2c')
               .add(name='res4b22')
               .relu(name='res4b22_relu'))

###################################### my densepose parsing head ######################################################
# !!!!!!! TODO
          # head_loss_gradients['body_uv'] = _add_roi_body_uv_head(
          #      model, add_roi_body_uv_head_func, blob_conv, dim_conv,
          #      spatial_scale_conv
          # )

# def add_roi_body_uv_head_v1convX(model, blob_in, dim_in, spatial_scale):
# I hardcoded configurable paraeters because it is easier to implement right now
          spatial_scale = 1. / 16.
          # blob_in = model # uvistvu pripne to zraven
          # dim_in = 
          # ^^^ i added this from code on github
          """v1convX design: X * (conv)."""
          hidden_dim = 512 #cfg.BODY_UV_RCNN.CONV_HEAD_DIM
          kernel_size = 3 #cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
          pad_size = kernel_size // 2
          
     
          # for i in range(8): #cfg.BODY_UV_RCNN.NUM_STACKED_CONVS
               # current = model.Conv(
               #      current,
               #      'body_conv_fcn' + str(i + 1),
               #      dim_in,
               #      hidden_dim,
               #      kernel_size,
               #      stride=1,
               #      pad=pad_size,
               #      weight_init=("MSRAFill" , {'std': 0.01}), #cfg.BODY_UV_RCNN.CONV_INIT
               #      bias_init=('ConstantFill', {'value': 0.})
               # )
               # current = model.Relu(current, current)
               # dim_in = hidden_dim

          # todo
          (self.feed('res4b22_relu', 'res4b22')
               .conv(3, 3, 512, 1, padding='SAME', biased=False, relu=True, name='body_conv_fcn1')
               .conv(3, 3, 512, 1, padding='SAME', biased=False, relu=True, name='body_conv_fcn2')
               .conv(3, 3, 512, 1, padding='SAME', biased=False, relu=True, name='body_conv_fcn3')
               .conv(3, 3, 512, 1, padding='SAME', biased=False, relu=True, name='body_conv_fcn4')
               .conv(3, 3, 512, 1, padding='SAME', biased=False, relu=True, name='body_conv_fcn5')
               .conv(3, 3, 512, 1, padding='SAME', biased=False, relu=True, name='body_conv_fcn6')
               .conv(3, 3, 512, 1, padding='SAME', biased=False, relu=True, name='body_conv_fcn7')
               .conv(3, 3, 512, 1, padding='SAME', biased=False, relu=True, name='body_conv_fcn8'))
     
###################################### end #####################################################

