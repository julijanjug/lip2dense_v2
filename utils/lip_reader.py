import os

import numpy as np
import tensorflow as tf
import random
import pandas as pd

import sys
import pickle
import json

IGNORE_LABEL = 255
NUM_POSE = 16
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def image_scaling(img, label, heatmap):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    
    scale = tf.random_uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
    heatmap = tf.image.resize_nearest_neighbor(heatmap, new_shape)
    return img, label, heatmap

def image_mirroring(img, label, label_rev, heatmap, heatmap_rev):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """

    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)

    flag = tf.less(distort_left_right_random, 0.5)
    mask = tf.stack([tf.logical_not(flag), flag])

    label_and_rev = tf.stack([label, label_rev])
    label_ = tf.boolean_mask(label_and_rev, mask)
    label_ = tf.reshape(label_, tf.shape(label))

    heatmap_and_rev = tf.stack([heatmap, heatmap_rev])
    heatmap_ = tf.boolean_mask(heatmap_and_rev, mask)
    heatmap_ = tf.reshape(heatmap_, tf.shape(heatmap))

    return img, label_, heatmap_

def random_resize_img_labels(image, label, heatmap, resized_h, resized_w):

    scale = tf.random_uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(resized_h), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(resized_w), scale))

    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(image, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
    heatmap = tf.image.resize_nearest_neighbor(tf.expand_dims(heatmap, 0), new_shape)
    heatmap = tf.squeeze(heatmap, squeeze_dims=[0])
    return img, label, heatmap

def resize_img_labels(image, label, heatmap, densepose, resized_h, resized_w):
    new_shape = tf.stack([tf.to_int32(resized_h), tf.to_int32(resized_w)])
    img = tf.image.resize(image, new_shape)
    label = tf.compat.v1.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
    new_shape_80 = tf.stack([tf.to_int32(resized_h / 8.0), tf.to_int32(resized_w / 8.0)])
    heatmap = tf.compat.v1.image.resize_nearest_neighbor(tf.expand_dims(heatmap, 0), new_shape_80)
    heatmap = tf.squeeze(heatmap, squeeze_dims=[0])

    #densepose resize
    # print('densepose ------------resize:  {}'.format(densepose))
    # image_height, image_width = image.shape[1], image.shape[0]
    # print('densepose ------------resize height: width:  {} -- {}'.format(image_height, image_width))
    # densepose_full = tf.image.pad_to_bounding_box(densepose, 0, 0, image_height, image_width) #replace 0 0 with y, x

    # densepose_resized = tf.image.resize_nearest_neighbor(tf.expand_dims(densepose_full, 0), new_shape)
    # densepose_resized = tf.squeeze(densepose_resized, squeeze_dims=[0])

    return img, label, heatmap, densepose

def random_crop_and_pad_image_and_labels(image, label, heatmap, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    heatmap = tf.cast(heatmap, dtype=tf.float32)
    combined = tf.concat([image, label, heatmap], 2) 
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))
    
    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4+NUM_POSE])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:last_image_dim+last_label_dim]
    heatmap_crop = combined_crop[:, :, last_image_dim+last_label_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    
    # Set static shape so that tensorflow knows shape at compile time. 
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    heatmap_crop.set_shape((crop_h, crop_w, NUM_POSE))
    new_shape = tf.stack([tf.to_int32(crop_h / 8.0), tf.to_int32(crop_w / 8.0)])
    heatmap = tf.image.resize_nearest_neighbor(tf.expand_dims(heatmap_crop, 0), new_shape)
    heatmap = tf.squeeze(heatmap, squeeze_dims=[0])
    return img_crop, label_crop, heatmap  


def read_labeled_image_list(dense_data_dir, data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    masks_rev = []
    denseposes = []
    for line in f:
        try:
            image, mask, mask_rev = line.strip("\n").split(' ')
        except ValueError: # Adhoc for test.
            image = mask = mask_rev = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
        masks_rev.append(data_dir + mask_rev)
        denseposes.append(dense_data_dir + '/formated-csv/' + image.split('.')[0].split('/')[2] + '.csv')
        # denseposes.append(dense_data_dir + image + '.txt')
    return images, masks, masks_rev, denseposes

def read_pose_list(data_dir, data_id_list):
    f = open(data_id_list, 'r')
    poses = []
    for line in f:
        pose = line.strip("\n")
        poses.append(data_dir + '/heatmap/' + pose)
    return poses

def read_from_csv(filename_queue):
  reader = tf.TextLineReader(skip_header_lines=0)
  scv_keys, csv_rows = reader.read_up_to(filename_queue, 128*128)
  record_defaults = [[0.0], [0.0], [0.0]]
  u, v, i = tf.decode_csv(csv_rows, record_defaults=record_defaults, field_delim=',')
  densepose = tf.stack([u,v,i])
  densepose.set_shape(tf.TensorShape([3, 128*128])) #TODO
  print("--------read from csv {}".format(densepose))
  return densepose

def read_images_from_disk(input_queue, dense_queue, input_size, random_scale, random_mirror=False): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      
    Returns:
      Two tensors: the decoded image and its mask. !* and densepose *!
    """

    img_contents = tf.io.read_file(input_queue[0])
    label_contents = tf.io.read_file(input_queue[1])
    label_contents_rev = tf.io.read_file(input_queue[2])
    densepose = read_from_csv(dense_queue)
    print("oeoepepep-------")
    print(densepose)

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    label = tf.image.decode_png(label_contents, channels=1)
    label_rev = tf.image.decode_png(label_contents_rev, channels=1)

    pose_id = input_queue[3]
    pose = []
    for i in range(NUM_POSE):
        pose_contents = tf.io.read_file(pose_id+'_{}.png'.format(i))
        pose_i = tf.image.decode_png(pose_contents, channels=1)
        pose.append(pose_i)
    heatmap = tf.concat(pose, axis=2)

    # create reversed heatmap
    pose_rev = [None] * 16
    pose_rev[0] = pose[5]
    pose_rev[1] = pose[4]
    pose_rev[2] = pose[3]
    pose_rev[3] = pose[2]
    pose_rev[4] = pose[1]
    pose_rev[5] = pose[0]
    pose_rev[10] = pose[15]
    pose_rev[11] = pose[14]
    pose_rev[12] = pose[13]
    pose_rev[13] = pose[12]
    pose_rev[14] = pose[11]
    pose_rev[15] = pose[10]
    pose_rev[6] = pose[6]
    pose_rev[7] = pose[7]
    pose_rev[8] = pose[8]
    pose_rev[9] = pose[9]
    heatmap_rev = tf.concat(pose_rev, axis=2)
    heatmap_rev = tf.reverse(heatmap_rev, tf.stack([1]))

    #Get densepose anotations
    # def parse_json(filename, input_size):
    #   if not tf.executing_eagerly():
    #     raise RuntimeError("TensorFlow must be executing eagerly.")
    #   print("PARSE_JSON file------- {}".format(filename))
    #   with open(filename) as f:
    #     tmp = f.read().replace("\\", "")[1:-1]
    #     js = json.loads(tmp)
    #     print("PARSE_JSON file------ pridel cez json loads")
    #     uvi = np.array([js['pred_densepose_uv'][0], js['pred_densepose_uv'][1], js['pred_densepose_labels']]) #u,v,labels
    #     print("PARSE_JSON file------ pridel cez uvi parsed del")
    #     print("PARSE_JSON file uvi ------ {}".format(uvi))
        
    #     return uvi

    # print("----------1")
    # [parsed] = tf.py_func(parse_json, [input_queue[4]], [tf.float32])
    # print("----------2")
    # h, w = input_size
    # parsed.set_shape(tf.TensorShape([h, w, 3]))
    # densepose = parsed

    #dense test csv 
    # def parse_csv(filename):
    #   print("PARSE_JSON file------- {}".format(filename))
    #   dense_test = pd.read_csv(filename, header=None)
    #   print("PARSE_JSON file 2 ------- {}".format(filename))
    #   dense_test_list = np.array(dense_test.values.tolist())
    #   dim = dense_test_list[4][0:2] #dim1 dim2

    #   test_u = dense_test_list[0].reshape((int(dim[0]), int(dim[1])))
    #   test_v = dense_test_list[1].reshape((int(dim[0]), int(dim[1])))
    #   test_i = dense_test_list[2].reshape((int(dim[0]), int(dim[1])))
    #   uvi = np.asarray([test_u, test_v, test_i], np.float32)
    #   uvi_tensor = tf.convert_to_tensor(uvi, np.float32)
    #   print("--------pandas dense  {}".format(uvi_tensor))  
    #   return uvi_tensor

    # [parsed] = tf.py_func(parse_csv, [input_queue[4]], [tf.float32])
    # h, w = input_size
    # parsed.set_shape(tf.TensorShape([h, w, 3]))
    # densepose = parsed

    if input_size is not None:
        h, w = input_size

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label, heatmap = image_mirroring(img, label, label_rev, heatmap, heatmap_rev)

        # Randomly resize the images and labels.
        if random_scale:
            img, label, heatmap = random_resize_img_labels(img, label, heatmap, h, w)
            # Random scale must be followed by crop to create fixed size
            img, label, heatmap = random_crop_and_pad_image_and_labels(img, label, heatmap, h, w, IGNORE_LABEL)
        else:
            img, label, heatmap, densepose = resize_img_labels(img, label, heatmap, densepose, h, w)

    print("-------------------------------------------------------------- konec load from disk" )
    print(densepose)
    print(img)
    return img, label, heatmap, densepose

class LIPReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, dense_data_dir, data_dir, data_list, data_id_list, input_size, random_scale,
                 random_mirror, shuffle, coord):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          data_id_list: path to the file of image id.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          coord: TensorFlow queue coordinator.
        '''
        self.dense_data_dir = dense_data_dir
        self.data_dir = data_dir
        self.data_list = data_list
        self.data_id_list = data_id_list
        self.input_size = input_size
        self.coord = coord


        self.image_list, self.label_list, self.label_rev_list, self.densepose_list = read_labeled_image_list(self.dense_data_dir, self.data_dir, self.data_list)
        self.pose_list = read_pose_list(self.data_dir, self.data_id_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.labels_rev = tf.convert_to_tensor(self.label_rev_list, dtype=tf.string)
        self.poses = tf.convert_to_tensor(self.pose_list, dtype=tf.string)
        self.denseposes = tf.convert_to_tensor(self.densepose_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels, self.labels_rev, self.poses, self.denseposes], shuffle=shuffle) 
        self.just_dense_queue = tf.train.string_input_producer(self.denseposes, shuffle=shuffle) 
        self.image, self.label, self.heatmap, self.densepose = read_images_from_disk(self.queue, self.just_dense_queue, self.input_size, random_scale, random_mirror) 

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        print("////////////////////////////// "+ str(num_elements))
        batch_list = [self.image, self.label, self.heatmap, self.densepose]
        print(self.densepose)
        print(self.image)
        image_batch, label_batch, heatmap_batch, densepose_batch = tf.train.batch([self.image, self.label, self.heatmap, self.densepose], num_elements)
        return image_batch, label_batch, heatmap_batch, densepose_batch
