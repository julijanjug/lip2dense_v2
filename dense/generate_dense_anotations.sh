#!/bin/bash

n=4

cd detectron2
source venv/bin/activate
cd projects/DensePose

python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl  ../../../LIP/LIP_dataset/train_set/images_1 --output ../../../LIP/anotations/dense_anotations/dense_train_1.pkl -v

python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl  ../../../LIP/LIP_dataset/train_set/images_2 --output ../../../LIP/anotations/dense_anotations/dense_train_2.pkl -v

python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl  ../../../LIP/LIP_dataset/train_set/images_3 --output ../../../LIP/anotations/dense_anotations/dense_train_3.pkl -v

python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl  ../../../LIP/LIP_dataset/train_set/images_4 --output ../../../LIP/anotations/dense_anotations/dense_train_4.pkl -v

python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl  ../../../LIP/LIP_dataset/train_set/images_5 --output ../../../LIP/anotations/dense_anotations/dense_train_5.pkl -v

python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl  ../../../LIP/LIP_dataset/train_set/images_6 --output ../../../LIP/anotations/dense_anotations/dense_train_6.pkl -v

python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl  ../../../LIP/LIP_dataset/train_set/images_7 --output ../../../LIP/anotations/dense_anotations/dense_train_7.pkl -v

python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl  ../../../LIP/LIP_dataset/train_set/images_8 --output ../../../LIP/anotations/dense_anotations/dense_train_8.pkl -v

python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl  ../../../LIP/LIP_dataset/train_set/images_9 --output ../../../LIP/anotations/dense_anotations/dense_train_9.pkl -v

python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl  ../../../LIP/LIP_dataset/train_set/images_10 --output ../../../LIP/anotations/dense_anotations/dense_train_10.pkl -v

python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl  ../../../LIP/LIP_dataset/train_set/images_11 --output ../../../LIP/anotations/dense_anotations/dense_train_11.pkl -v


