import numpy as np
import tensorflow as tf
import os
import scipy.misc

#this functions i needed to convert from pytorch syntax and apply it to lip data
#za podrobnosti glej densepose/modeling/predictors/chart

def process_dense_head_output(head_output):
    dim_in = 24 # input_channels
    n_segm_chan = 2 #cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
    dim_out_patches = 24+1 #cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
    kernel_size = 4 #cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
    scale_factor = 2 #cfg.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE

    # vsi spodnji so bili enako poracunani 
    #pretvorba je bila nareta po tem: https://datascience.stackexchange.com/questions/26451/how-to-calculate-the-output-shape-of-conv2d-transpose
    # coarse segmentation
    # self.ann_index_lowres = ConvTranspose2d( 
    #     dim_in, n_segm_chan, kernel_size, stride=2, padding=int(kernel_size / 2 - 1))
    print("--------head_output 123--- {}".format(head_output))
    # with tf.variable_scope('', reuse=True):
    ann_index_lowres = tf.compat.v1.layers.Conv2DTranspose(dim_out_patches, [kernel_size, kernel_size], strides=2, padding="VALID")
    # fine segmentation
    index_uv_lowres = tf.compat.v1.layers.Conv2DTranspose(dim_out_patches, [kernel_size, kernel_size], strides=2, padding="VALID")
    # U
    u_lowres = tf.compat.v1.layers.Conv2DTranspose(dim_out_patches, [kernel_size, kernel_size], strides=2, padding="VALID")
    # V
    v_lowres = tf.compat.v1.layers.Conv2DTranspose(dim_out_patches, [kernel_size, kernel_size], strides=2, padding="VALID")
    print("--------head_output i--- {}".format(index_uv_lowres))
    print("--------head_output u--- {}".format(u_lowres))
    print("--------head_output v--- {}".format(v_lowres))

    # return interp2d(ann_index_lowres(head_output))

    return tf.stack([interp2d(ann_index_lowres(head_output)), interp2d(index_uv_lowres(head_output)), interp2d(u_lowres(head_output)), interp2d(v_lowres(head_output))])

def interp2d(image):
    """
    Bilinear interpolation method to be used for upscaling

    Args:
        tensor_nchw (tensor): tensor of shape (N, C, H, W)
    Return:
        tensor of shape (N, C, Hout, Wout), where Hout and Wout are computed
            by applying the scale factor to H and W
    """
    # return interpolate(
    #     tensor_nchw, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
    # ) 
    size = [128, 128]
    # to funkcijo bo treba dodaelat
    return tf.image.resize(image, size)
