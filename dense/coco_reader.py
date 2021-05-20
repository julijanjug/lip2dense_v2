import os

import numpy as np
import tensorflow as tf
import random

from pycocotools.coco import COCO
from PIL import Image

DATA_TYPE = '2014_train'
ANNOTATION_FILE = '../COCO/annotations/densepose_coco_{}.json'.format(DATA_TYPE)
CATEGORY_CLASSES = ['person'] # only persons
coco = COCO(ANNOTATION_FILE)


IGNORE_LABEL = 255
NUM_POSE = 16
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def resize_img_labels(image, label, heatmap, densepose, resized_h, resized_w):

  new_shape = tf.stack([tf.to_int32(resized_h), tf.to_int32(resized_w)])
  print("---------------newshape/image/label--{}--{}-{}".format(new_shape, image, label.shape))
  img = tf.image.resize_images(image, new_shape)
  label = tf.image.resize_nearest_neighbor(tf.expand_dims(tf.expand_dims(label, 0), -1), new_shape)
  label = tf.squeeze(label, squeeze_dims=[0])
  new_shape = tf.stack([tf.to_int32(resized_h / 8.0), tf.to_int32(resized_w / 8.0)])
  # heatmap = tf.image.resize_nearest_neighbor(tf.expand_dims(heatmap, 0), new_shape)
  # heatmap = tf.squeeze(heatmap, squeeze_dims=[0])
  return img, label, heatmap, densepose

def read_labeled_image_list(data_dir, data_type):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    *!* I added dataType for custom loading from coco
    And it now returns an list of image informations like image id and path 

    Returns:
      Two lists with all file names for images and masks, respectively.
    """

    cat_ids = coco.getCatIds(catNms=CATEGORY_CLASSES)
    ann_ids = coco.getAnnIds(catIds= cat_ids)

    return ann_ids


def read_images_from_disk(input_queue, input_size, random_scale, data_dir, random_mirror=False): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask. Annotations
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """

    ann = coco.loadAnns(input_queue[0])[0] #input queue je seznami idjev vzeli bomo zaenkrat samo 1 ann iz slike
    image = coco.loadImgs(ann["image_id"])[0] #nalozimo json ki opisuje podatke o sliki
    print("_______ann/image/input_queue[0]-{}--{}---{}".format(ann, image, input_queue[0]))
    x,y,w,h = int(ann['bbox'][0]), int(ann['bbox'][1]), int(ann['bbox'][2]), int(ann['bbox'][3])
    img_contents = tf.read_file(data_dir + image["file_name"])
    # img_contents = Image.open(data_dir + image["file_name"])
    
    # img_contents = tf.read_file(input_queue[0])
    # label_contents = tf.read_file(input_queue[1])
    # label_contents_rev = tf.read_file(input_queue[2])
    
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img = tf.image.crop_to_bounding_box(img, y,x,h,w)
    # print(img_contents.size)
    # print("{} {} {} {}".format(x,y,w,h))
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    # label = tf.image.decode_png(label_contents, channels=1)
    # label_rev = tf.image.decode_png(label_contents_rev, channels=1)

    label = coco.annToMask(ann) #ne vem ali je potrebno podat samo id al cel anotation
    heatmap = 0 #TODO potrebno je generirat heatmape za pozicije in jih nalozit tu
    densepose = [ann['dp_I'], ann['dp_V'], ann['dp_U'], ann['dp_x'], ann['dp_y']]

    # pose_id = input_queue[3]
    # pose = []
    # for i in range(NUM_POSE):
    #     pose_contents = tf.read_file(pose_id+'_{}.png'.format(i))
    #     pose_i = tf.image.decode_png(pose_contents, channels=1)
    #     pose.append(pose_i)
    # heatmap = tf.concat(pose, axis=2)

    if input_size is not None:
        h, w = input_size #!!! potrebno je dejansko resizat labele ce je podan size

        # # Randomly mirror the images and labels.
        # if random_mirror:
        #     img, label, heatmap = image_mirroring(img, label, label_rev, heatmap, heatmap_rev)

        # # Randomly resize the images and labels.
        # if random_scale:
        #     img, label, heatmap = random_resize_img_labels(img, label, heatmap, h, w)
        #     # Random scale must be followed by crop to create fixed size
        #     img, label, heatmap = random_crop_and_pad_image_and_labels(img, label, heatmap, h, w, IGNORE_LABEL)
        # else:
        #     img, label, heatmap = resize_img_labels(img, label, heatmap, h, w)

        img, label, heatmap, densepose = resize_img_labels(img, label, heatmap, densepose, h, w)

    return img, label, heatmap, densepose

class COCOReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_type, anno_dir, input_size, random_scale,
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

        self.data_dir = data_dir
        self.anno_dir = anno_dir
        self.data_type = data_type
        self.input_size = input_size
        self.coord = coord

        self.image_list = read_labeled_image_list(self.data_dir, self.data_type) # list of annotation ids
        # self.images = tf.convert_to_tensor(self.image_list, dtype=tf.int32)
        self.images = self.image_list

        #todo read correct files 
        self.queue = tf.train.slice_input_producer([self.images], shuffle=shuffle) 
        self.image, self.label, self.heatmap , self.densepose = read_images_from_disk(self.images, self.input_size, random_scale, self.data_dir, random_mirror) #prej se je notr namesto images poslal queue

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {4, 1}) for images and masks.'''
        batch_list = [self.image, self.label, self.heatmap, self.densepose]
        image_batch, label_batch, heatmap_batch, densepose_batch = tf.train.batch([self.image, self.label, self.heatmap, self.densepose], num_elements)
        return image_batch, label_batch, heatmap_batch, densepose_batch
