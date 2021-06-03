from __future__ import print_function
import os
import time
import tensorflow as tf
import numpy as np
import random
from utils import *
from LIP_model import *

from statistics import mean 

from dense import *
from dense.dense_model import *
from dense.dense_utils import *
from dense.utils import *

# Set gpus
gpus = [0] # Here I set CUDA to only see one GPU
os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in gpus])
num_gpus = len(gpus) # number of GPUs to use

### parameters setting
N_CLASSES = 20
DENSE_N_CLASSES = 24
# INPUT_SIZE = (384, 384)
#  I reduced the image size for faster procesing
INPUT_SIZE = (128, 128)

BATCH_SIZE = 18
BATCH_I = 2
SHUFFLE = False
RANDOM_SCALE = False
RANDOM_MIRROR = False
LEARNING_RATE = 2e-4
MOMENTUM = 0.9
POWER = 0.9
NUM_STEPS = 7616 * 35 + 1
SAVE_PRED_EVERY = 7616 
p_Weight = 1 #parsing
s_Weight = 1 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! nastavil sem da se loss uposteva samo od parsing dela
d_Weight = 1 #dense pose loss weight
DATA_DIR = './datasets/lip/LIP_dataset/train_set'
LIST_PATH = './datasets/lip/list/train_rev.txt'
DATA_ID_LIST = './datasets/lip/list/train_id.txt'
DENSE_ANN_DIR= '../LIP/anotations/dense_anotations/train'

SNAPSHOT_DIR = './checkpoint/lip2dense_v2'
LOG_DIR = './logs/lip2dense_v2'


def main():
    RANDOM_SEED = random.randint(1000, 9999)
    tf.set_random_seed(RANDOM_SEED)

    # Create queue coordinator.
    coord = tf.train.Coordinator()
    h, w = INPUT_SIZE
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = LIPReader(DENSE_ANN_DIR, DATA_DIR, LIST_PATH, DATA_ID_LIST, INPUT_SIZE, RANDOM_SCALE, RANDOM_MIRROR, SHUFFLE, coord)
        image_batch, label_batch, heatmap_batch, densepose_batch = reader.dequeue(BATCH_SIZE)
        image_batch075 = tf.image.resize_images(image_batch, [int(h * 0.75), int(w * 0.75)])
        image_batch050 = tf.image.resize_images(image_batch, [int(h * 0.5), int(w * 0.5)])
        heatmap_batch = tf.scalar_mul(1.0/255, tf.cast(heatmap_batch, tf.float32))

    tower_grads = []
    reuse1 = False
    reuse2 = False
    # Define loss and optimisation parameters.
    base_lr = tf.constant(LEARNING_RATE)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / NUM_STEPS), POWER))
    optim = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)

    for i in range (num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('Tower_%d' % (i)) as scope:
                if i == 0:
                    reuse1 = False
                    reuse2 = True
                else:
                    reuse1 = True
                    reuse2 = True
                next_image = image_batch[i*BATCH_I:(i+1)*BATCH_I,:]
                next_image075 = image_batch075[i*BATCH_I:(i+1)*BATCH_I,:]
                next_image050 = image_batch050[i*BATCH_I:(i+1)*BATCH_I,:]
                next_heatmap = heatmap_batch[i*BATCH_I:(i+1)*BATCH_I,:]
                next_label = label_batch[i*BATCH_I:(i+1)*BATCH_I,:]
                next_densepose_label = densepose_batch[i*BATCH_I:(i+1)*BATCH_I,:]
                print("-----nesxt densepose label: {}".format(next_densepose_label))

                # Create network.
                print("________next_image{}".format(next_image))
                print("--------image_batch--{}".format(image_batch))
                with tf.variable_scope('', reuse=reuse1):
                    net_100 = JPPNetModel({'data': next_image}, is_training=False, n_classes=N_CLASSES)
                with tf.variable_scope('', reuse=reuse2):
                    net_075 = JPPNetModel({'data': next_image075}, is_training=False, n_classes=N_CLASSES)
                with tf.variable_scope('', reuse=reuse2):
                    net_050 = JPPNetModel({'data': next_image050}, is_training=False, n_classes=N_CLASSES)

                # parsing net
                parsing_fea1_100 = net_100.layers['res5d_branch2b_parsing']
                parsing_fea1_075 = net_075.layers['res5d_branch2b_parsing']
                parsing_fea1_050 = net_050.layers['res5d_branch2b_parsing']

                parsing_out1_100 = net_100.layers['fc1_human']
                print("--------parsing_out1_100--{}".format(parsing_out1_100))
                parsing_out1_075 = net_075.layers['fc1_human']
                parsing_out1_050 = net_050.layers['fc1_human']
                # pose net
                resnet_fea_100 = net_100.layers['res4b22_relu']
                resnet_fea_075 = net_075.layers['res4b22_relu']
                resnet_fea_050 = net_050.layers['res4b22_relu']
                #!!! densepose net
                densepose_fea_100 = net_100.layers['res4b22_relu']
                densepose_fea_075 = net_100.layers['res4b22_relu']
                densepose_fea_050 = net_100.layers['res4b22_relu']
                
                with tf.variable_scope('', reuse=reuse1):
                    dense_out1_100, dense_fea1_100 = dense_net(densepose_fea_100, 'fc1_dense') # custom densepose glava

                    pose_out1_100, pose_fea1_100 = pose_net(resnet_fea_100, 'fc1_pose')
                    pose_out2_100, pose_fea2_100 = pose_refine(pose_out1_100, parsing_out1_100, pose_fea1_100, name='fc2_pose')
                    parsing_out2_100, parsing_fea2_100 = parsing_refine(parsing_out1_100, pose_out1_100, parsing_fea1_100, name='fc2_parsing')
                    parsing_out3_100, parsing_fea3_100 = parsing_refine(parsing_out2_100, pose_out2_100, parsing_fea2_100, name='fc3_parsing')
                    pose_out3_100, pose_fea3_100 = pose_refine(pose_out2_100, parsing_out2_100, pose_fea2_100, name='fc3_pose')

                with tf.variable_scope('', reuse=reuse2):
                    dense_out1_075, dense_fea1_075 = dense_net(densepose_fea_075, 'fc1_dense') # custom densepose glava

                    pose_out1_075, pose_fea1_075 = pose_net(resnet_fea_075, 'fc1_pose')
                    pose_out2_075, pose_fea2_075 = pose_refine(pose_out1_075, parsing_out1_075, pose_fea1_075, name='fc2_pose')
                    parsing_out2_075, parsing_fea2_075 = parsing_refine(parsing_out1_075, pose_out1_075, parsing_fea1_075, name='fc2_parsing')
                    parsing_out3_075, parsing_fea3_075 = parsing_refine(parsing_out2_075, pose_out2_075, parsing_fea2_075, name='fc3_parsing')
                    pose_out3_075, pose_fea3_075 = pose_refine(pose_out2_075, parsing_out2_075, pose_fea2_075, name='fc3_pose')

                with tf.variable_scope('', reuse=reuse2):
                    dense_out1_050, dense_fea1_050 = dense_net(densepose_fea_050, 'fc1_dense') # custom densepose glava

                    pose_out1_050, pose_fea1_050 = pose_net(resnet_fea_050, 'fc1_pose')
                    pose_out2_050, pose_fea2_050 = pose_refine(pose_out1_050, parsing_out1_050, pose_fea1_050, name='fc2_pose')
                    parsing_out2_050, parsing_fea2_050 = parsing_refine(parsing_out1_050, pose_out1_050, parsing_fea1_050, name='fc2_parsing')
                    parsing_out3_050, parsing_fea3_050 = parsing_refine(parsing_out2_050, pose_out2_050, parsing_fea2_050, name='fc3_parsing')
                    pose_out3_050, pose_fea3_050 = pose_refine(pose_out2_050, parsing_out2_050, pose_fea2_050, name='fc3_pose')

                # combine resize
                parsing_out1 = tf.reduce_mean(tf.stack([parsing_out1_100,
                                                     tf.image.resize_images(parsing_out1_075, tf.shape(parsing_out1_100)[1:3,]),
                                                     tf.image.resize_images(parsing_out1_050, tf.shape(parsing_out1_100)[1:3,])]), axis=0)
                parsing_out2 = tf.reduce_mean(tf.stack([parsing_out2_100,
                                                     tf.image.resize_images(parsing_out2_075, tf.shape(parsing_out2_100)[1:3,]),
                                                     tf.image.resize_images(parsing_out2_050, tf.shape(parsing_out2_100)[1:3,])]), axis=0)
                parsing_out3 = tf.reduce_mean(tf.stack([parsing_out3_100,
                                                     tf.image.resize_images(parsing_out3_075, tf.shape(parsing_out3_100)[1:3,]),
                                                     tf.image.resize_images(parsing_out3_050, tf.shape(parsing_out3_100)[1:3,])]), axis=0)
                pose_out1 = tf.reduce_mean(tf.stack([pose_out1_100,
                                                     tf.image.resize_nearest_neighbor(pose_out1_075, tf.shape(pose_out1_100)[1:3,]),
                                                     tf.image.resize_nearest_neighbor(pose_out1_050, tf.shape(pose_out1_100)[1:3,])]), axis=0)
                pose_out2 = tf.reduce_mean(tf.stack([pose_out2_100,
                                                     tf.image.resize_nearest_neighbor(pose_out2_075, tf.shape(pose_out2_100)[1:3,]),
                                                     tf.image.resize_nearest_neighbor(pose_out2_050, tf.shape(pose_out2_100)[1:3,])]), axis=0)
                pose_out3 = tf.reduce_mean(tf.stack([pose_out3_100,
                                                     tf.image.resize_nearest_neighbor(pose_out3_075, tf.shape(pose_out3_100)[1:3,]),
                                                     tf.image.resize_nearest_neighbor(pose_out3_050, tf.shape(pose_out3_100)[1:3,])]), axis=0)
                dense_out1 = tf.reduce_mean(tf.stack([dense_out1_100,
                                                     tf.image.resize_images(dense_out1_075, tf.shape(dense_out1_100)[1:3,]),
                                                     tf.image.resize_images(dense_out1_050, tf.shape(dense_out1_100)[1:3,])]), axis=0)

                ### Predictions: ignoring all predictions with labels greater or equal than n_classes
                raw_prediction_p1 = tf.reshape(parsing_out1, [-1, N_CLASSES])
                raw_prediction_p1_100 = tf.reshape(parsing_out1_100, [-1, N_CLASSES])
                raw_prediction_p1_075 = tf.reshape(parsing_out1_075, [-1, N_CLASSES])
                raw_prediction_p1_050 = tf.reshape(parsing_out1_050, [-1, N_CLASSES])

                raw_prediction_p2 = tf.reshape(parsing_out2, [-1, N_CLASSES])
                raw_prediction_p2_100 = tf.reshape(parsing_out2_100, [-1, N_CLASSES])
                raw_prediction_p2_075 = tf.reshape(parsing_out2_075, [-1, N_CLASSES])
                raw_prediction_p2_050 = tf.reshape(parsing_out2_050, [-1, N_CLASSES])

                raw_prediction_p3 = tf.reshape(parsing_out3, [-1, N_CLASSES])
                raw_prediction_p3_100 = tf.reshape(parsing_out3_100, [-1, N_CLASSES])
                raw_prediction_p3_075 = tf.reshape(parsing_out3_075, [-1, N_CLASSES])
                raw_prediction_p3_050 = tf.reshape(parsing_out3_050, [-1, N_CLASSES])

                print("--------parsing_out1 {}".format(parsing_out1))
                label_proc = prepare_label(next_label, tf.stack(parsing_out1.get_shape()[1:3]), one_hot=False) # [batch_size, h, w]
                label_proc075 = prepare_label(next_label, tf.stack(parsing_out1_075.get_shape()[1:3]), one_hot=False)
                label_proc050 = prepare_label(next_label, tf.stack(parsing_out1_050.get_shape()[1:3]), one_hot=False)

                raw_gt = tf.reshape(label_proc, [-1,])
                raw_gt075 = tf.reshape(label_proc075, [-1,])
                raw_gt050 = tf.reshape(label_proc050, [-1,])

                indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, N_CLASSES - 1)), 1)
                indices075 = tf.squeeze(tf.where(tf.less_equal(raw_gt075, N_CLASSES - 1)), 1)
                indices050 = tf.squeeze(tf.where(tf.less_equal(raw_gt050, N_CLASSES - 1)), 1)

                gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
                gt075 = tf.cast(tf.gather(raw_gt075, indices075), tf.int32)
                gt050 = tf.cast(tf.gather(raw_gt050, indices050), tf.int32)

                #TODO this needs to be worked on urgently
                #raw dense predictions !!!
                print("--------dense_out1--- {}".format(dense_out1))
                processed_dense_head_outputs = process_dense_head_output(dense_out1)
                processed_dense_head_outputs_100 = process_dense_head_output(dense_out1_100)
                processed_dense_head_outputs_075 = process_dense_head_output(dense_out1_075)
                processed_dense_head_outputs_050 = process_dense_head_output(dense_out1_050)
                
                u_lowers_pred = tf.gather(processed_dense_head_outputs, [0])
                v_lowres_pred = tf.gather(processed_dense_head_outputs, [1])
                i_lowres_pred = tf.gather(processed_dense_head_outputs, [2])
                u_lowers_pred_100 = tf.gather(processed_dense_head_outputs_100, [0])
                v_lowres_pred_100 = tf.gather(processed_dense_head_outputs_100, [1])
                i_lowres_pred_100 = tf.gather(processed_dense_head_outputs_100, [2])
                u_lowers_pred_075 = tf.gather(processed_dense_head_outputs_075, [0])
                v_lowres_pred_075 = tf.gather(processed_dense_head_outputs_075, [1])
                i_lowres_pred_075 = tf.gather(processed_dense_head_outputs_075, [2])                
                u_lowers_pred_050 = tf.gather(processed_dense_head_outputs_050, [0])
                v_lowres_pred_050 = tf.gather(processed_dense_head_outputs_050, [1])
                i_lowres_pred_050 = tf.gather(processed_dense_head_outputs_050, [2])

                #prepare next densepose label
                print("--------next_densepose_label--- {}".format(next_densepose_label))
                dense_label_proc = prepare_dense_label(next_densepose_label, tf.stack(dense_out1.get_shape()[1:]), one_hot=False) # [batch_size, h, w] se nekaj je 4 dimenzija
                dense_label_proc075 = prepare_dense_label(next_densepose_label, tf.stack(dense_out1_075.get_shape()[1:]), one_hot=False)
                dense_label_proc050 = prepare_dense_label(next_densepose_label, tf.stack(dense_out1_050.get_shape()[1:]), one_hot=False)

                #spremenimo dense labele v vektorje
                print("-----------dense_label_proc--{}".format(dense_label_proc))
                raw_dense_gt = tf.reshape(dense_label_proc, [-1,]) #spremeni labele v vektor
                raw_dense_gt075 = tf.reshape(dense_label_proc075, [-1,])
                raw_dense_gt050 = tf.reshape(dense_label_proc050, [-1,])

                prediction_p1 = tf.gather(raw_prediction_p1, indices)
                prediction_p1_100 = tf.gather(raw_prediction_p1_100, indices)
                prediction_p1_075 = tf.gather(raw_prediction_p1_075, indices075)
                prediction_p1_050 = tf.gather(raw_prediction_p1_050, indices050)

                prediction_p2 = tf.gather(raw_prediction_p2, indices)
                prediction_p2_100 = tf.gather(raw_prediction_p2_100, indices)
                prediction_p2_075 = tf.gather(raw_prediction_p2_075, indices075)
                prediction_p2_050 = tf.gather(raw_prediction_p2_050, indices050)

                prediction_p3 = tf.gather(raw_prediction_p3, indices)
                prediction_p3_100 = tf.gather(raw_prediction_p3_100, indices)
                prediction_p3_075 = tf.gather(raw_prediction_p3_075, indices075)
                prediction_p3_050 = tf.gather(raw_prediction_p3_050, indices050)

                next_heatmap075 = tf.image.resize_nearest_neighbor(next_heatmap, pose_out1_075.get_shape()[1:3])
                next_heatmap050 = tf.image.resize_nearest_neighbor(next_heatmap, pose_out1_050.get_shape()[1:3])

                ### Pixel-wise softmax loss.
                loss_p1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p1, labels=gt))
                loss_p1_100 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p1_100, labels=gt))
                loss_p1_075 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p1_075, labels=gt075))
                loss_p1_050 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p1_050, labels=gt050))

                loss_p2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p2, labels=gt))
                loss_p2_100 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p2_100, labels=gt))
                loss_p2_075 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p2_075, labels=gt075))
                loss_p2_050 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p2_050, labels=gt050))

                loss_p3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p3, labels=gt))
                loss_p3_100 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p3_100, labels=gt))
                loss_p3_075 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p3_075, labels=gt075))
                loss_p3_050 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_p3_050, labels=gt050))

                loss_s1 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, pose_out1)), [1, 2, 3])))
                loss_s1_100 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, pose_out1_100)), [1, 2, 3])))
                loss_s1_075 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap075, pose_out1_075)), [1, 2, 3])))
                loss_s1_050 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap050, pose_out1_050)), [1, 2, 3])))

                loss_s2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, pose_out2)), [1, 2, 3])))
                loss_s2_100 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, pose_out2_100)), [1, 2, 3])))
                loss_s2_075 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap075, pose_out2_075)), [1, 2, 3])))
                loss_s2_050 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap050, pose_out2_050)), [1, 2, 3])))

                loss_s3 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, pose_out3)), [1, 2, 3])))
                loss_s3_100 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, pose_out3_100)), [1, 2, 3])))
                loss_s3_075 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap075, pose_out3_075)), [1, 2, 3])))
                loss_s3_050 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap050, pose_out3_050)), [1, 2, 3])))

                #Densepose loss TODO finish
                print("densepose loss parts : {} -- {}".format(i_lowres_pred, raw_dense_gt))
                loss_d1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=i_lowres_pred, labels=raw_dense_gt)) #izracuna loss za densepose predikcije 1
                loss_d1_100 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=i_lowres_pred_100, labels=raw_dense_gt))
                loss_d1_075 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=i_lowres_pred_075, labels=raw_dense_gt075))
                loss_d1_050 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=i_lowres_pred_050, labels=raw_dense_gt050))

                loss_d2 = 0  # izracuna loss za densepose predikcije 2
                loss_d2_100 = 0
                loss_d2_075 = 0
                loss_d2_050 = 0

                loss_d3 = 0 # izracuna loss za densepose predikcije 3
                loss_d3_100 = 0
                loss_d3_075 = 0
                loss_d3_050 = 0

                loss_parsing = loss_p1 + loss_p1_100 + loss_p1_075 + loss_p1_050 + loss_p2 + loss_p2_100 + loss_p2_075 + loss_p2_050 + loss_p3 + loss_p3_100 + loss_p3_075 + loss_p3_050
                loss_pose = loss_s1 + loss_s1_100 + loss_s1_075 + loss_s1_050 + loss_s2 + loss_s2_100 + loss_s2_075 + loss_s2_050 + loss_s3 + loss_s3_100 + loss_s3_075 + loss_s3_050
                loss_densepose = loss_d1 + loss_d1_100 + loss_d1_075 + loss_d1_050 + loss_d2 + loss_d2_100 + loss_d2_075 + loss_d2_050 + loss_d3 + loss_d3_100 + loss_d3_075 + loss_d3_050
                reduced_loss =  loss_pose * s_Weight + loss_parsing * p_Weight + loss_densepose * d_Weight

                trainable_variable = tf.compat.v1.trainable_variables()
                grads = optim.compute_gradients(reduced_loss, var_list=trainable_variable)
                
                tower_grads.append(grads)

                tf.compat.v1.add_to_collection('loss_p1', loss_p1)
                tf.compat.v1.add_to_collection('loss_p2', loss_p2)
                tf.compat.v1.add_to_collection('loss_p3', loss_p3)
                tf.compat.v1.add_to_collection('loss_s1', loss_s1)
                tf.compat.v1.add_to_collection('loss_s2', loss_s2)
                tf.compat.v1.add_to_collection('loss_s3', loss_s3)
                tf.compat.v1.add_to_collection('reduced_loss', reduced_loss)

    # Average the gradients
    grads_ave = average_gradients(tower_grads)
    # apply the gradients with our optimizers
    train_op = optim.apply_gradients(grads_ave)

    loss_p1_ave = tf.reduce_mean(tf.get_collection('loss_p1'))
    loss_p2_ave = tf.reduce_mean(tf.get_collection('loss_p2'))
    loss_p3_ave = tf.reduce_mean(tf.get_collection('loss_p3'))
    loss_s1_ave = tf.reduce_mean(tf.get_collection('loss_s1'))
    loss_s2_ave = tf.reduce_mean(tf.get_collection('loss_s2'))
    loss_s3_ave = tf.reduce_mean(tf.get_collection('loss_s3'))
    loss_ave = tf.reduce_mean(tf.get_collection('reduced_loss'))

    loss_summary_p1 = tf.summary.scalar("loss_p1_ave", loss_p1_ave)
    loss_summary_p2 = tf.summary.scalar("loss_p2_ave", loss_p2_ave)
    loss_summary_p3 = tf.summary.scalar("loss_p3_ave", loss_p3_ave)
    loss_summary_s1 = tf.summary.scalar("loss_s1_ave", loss_s1_ave)
    loss_summary_s2 = tf.summary.scalar("loss_s2_ave", loss_s2_ave)
    loss_summary_s3 = tf.summary.scalar("loss_s3_ave", loss_s3_ave)
    loss_summary_ave = tf.summary.scalar("loss_ave", loss_ave)
    loss_summary = tf.summary.merge([loss_summary_ave, loss_summary_s1, loss_summary_s2, loss_summary_s3, loss_summary_p1, loss_summary_p2, loss_summary_p3])
    summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

    # Set up tf session and initialize variables.
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    all_saver_var = tf.compat.v1.global_variables()
    restore_var = all_saver_var #[v for v in all_saver_var if 'pose' not in v.name and 'parsing' not in v.name]
    saver = tf.compat.v1.train.Saver(var_list=all_saver_var, max_to_keep=50)
    loader = tf.compat.v1.train.Saver(var_list=restore_var)

    if load(loader, sess, SNAPSHOT_DIR):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")    

    # Start queue threads.
    threads = tf.compat.v1.train.start_queue_runners(coord=coord, sess=sess)

    # create a loss stats file    
    lossFile = open(SNAPSHOT_DIR + "/loss_stats.txt", "a+") 
    runningLoss = [0 for i in range(50)]

    # Iterate over training steps.
    for step in range(NUM_STEPS):
        start_time = time.time()
        loss_value = 0
        feed_dict = { step_ph : step }

        # Apply gradients.
        summary, loss_value, _ = sess.run([loss_summary, reduced_loss, train_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary, step)
        if step % SAVE_PRED_EVERY == 0:
            save(saver, sess, SNAPSHOT_DIR, step)
        
        # loss calculations
        runningLoss.append(loss_value)
        runningLoss.pop(0)
        if step % 500 == 0:
            lossFile.write('\n Step {0:10}:  {1:<10.5}'.format(str(step), mean(runningLoss)))

        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      
      if g is not None:
        expanded_g = tf.expand_dims(g, 0) # TODO FIX - it is somethimes None

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

    # Average over the 'tower' dimension.
    # print("-----------Grads-- {}".format(grads))
    if  len(grads) > 0:
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
  return average_grads

if __name__ == '__main__':
    main()
