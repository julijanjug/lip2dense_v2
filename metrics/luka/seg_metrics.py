import os
import numpy as np
from PIL import Image, ImageDraw

RGB_masks_SCHP = {
	"background": [0,0,0],
	"hat": [128,0,0],
	"hair": [0,128,0],
	"glove": [128,128,0],
	"sunglasses": [0,0,128],
	"upperclothes": [128,0,128],
	"dress": [0,128,128],
	"coat": [128,128,128],
	"socks": [64,0,0],
	"pants": [192,0,0],
	"jumpsuits": [64,128,0],
	"scarf": [192,128,0],
	"skirt": [64,0,128],
	"face": [192,0,128],
	"left-arm": [64,128,128],
	"right-arm": [192,128,128],
	"left-leg": [0,64,0],
	"right-leg": [128,64,0],
	"left-shoe": [0,192,0],
	"right-shoe": [128,192,0]
} 

masks_GT = {
	"background": [0,0,0],

	"hat": [1,1,1],

	"hair": [2,2,2],
	"sunglasses": [4,4,4],
	"face": [13,13,13],

	"scarf": [11,11,11],

	"glove": [3,3,3],
	"left-arm": [14,14,14],
	"right-arm": [15,15,15],

	"coat": [7,7,7],
	"upperclothes": [5,5,5],
	"dress": [6,6,6],

	"jumpsuits": [10,10,10],

	"pants": [9,9,9],
	"skirt": [12,12,12],

	"left-leg": [16,16,16],
	"right-leg": [17,17,17],

	"socks": [8,8,8],
	"left-shoe": [18,18,18],
	"right-shoe": [19,19,19]
} 

garment_IOU = {
	"background": 0,
	"hat": 0,
	"hair": 0,
	"glove": 0,
	"sunglasses": 0,
	"upperclothes": 0,
	"dress": 0,
	"coat": 0,
	"socks": 0,
	"pants": 0,
	"jumpsuits": 0,
	"scarf": 0,
	"skirt": 0,
	"face": 0,
	"left-arm": 0,
	"right-arm": 0,
	"left-leg": 0,
	"right-leg": 0,
	"left-shoe": 0,
	"right-shoe": 0
} 

garment_precision = {
	"background": 0,
	"hat": 0,
	"hair": 0,
	"glove": 0,
	"sunglasses": 0,
	"upperclothes": 0,
	"dress": 0,
	"coat": 0,
	"socks": 0,
	"pants": 0,
	"jumpsuits": 0,
	"scarf": 0,
	"skirt": 0,
	"face": 0,
	"left-arm": 0,
	"right-arm": 0,
	"left-leg": 0,
	"right-leg": 0,
	"left-shoe": 0,
	"right-shoe": 0
} 

garment_recall = {
	"background": 0,
	"hat": 0,
	"hair": 0,
	"glove": 0,
	"sunglasses": 0,
	"upperclothes": 0,
	"dress": 0,
	"coat": 0,
	"socks": 0,
	"pants": 0,
	"jumpsuits": 0,
	"scarf": 0,
	"skirt": 0,
	"face": 0,
	"left-arm": 0,
	"right-arm": 0,
	"left-leg": 0,
	"right-leg": 0,
	"left-shoe": 0,
	"right-shoe": 0
} 

garment_f1_score = {
	"background": 0,
	"hat": 0,
	"hair": 0,
	"glove": 0,
	"sunglasses": 0,
	"upperclothes": 0,
	"dress": 0,
	"coat": 0,
	"socks": 0,
	"pants": 0,
	"jumpsuits": 0,
	"scarf": 0,
	"skirt": 0,
	"face": 0,
	"left-arm": 0,
	"right-arm": 0,
	"left-leg": 0,
	"right-leg": 0,
	"left-shoe": 0,
	"right-shoe": 0
} 

garment_count = {
	"background": 0,
	"hat": 0,
	"hair": 0,
	"glove": 0,
	"sunglasses": 0,
	"upperclothes": 0,
	"dress": 0,
	"coat": 0,
	"socks": 0,
	"pants": 0,
	"jumpsuits": 0,
	"scarf": 0,
	"skirt": 0,
	"face": 0,
	"left-arm": 0,
	"right-arm": 0,
	"left-leg": 0,
	"right-leg": 0,
	"left-shoe": 0,
	"right-shoe": 0
} 

#DATA PATHS
# pred_dir = '/home/lukak/workspace/datasets/seg_result/lip/hrnet_w48_ocr_1_val/label/'
#pred_dir = '/home/lukak/workspace/datasets/lip/lip_JPPNet_predictions/' 
# pred_dir = '/media/sda1/JJug/mag/LIP_JPPNet/output/parsing/val/' 
pred_dir = '/media/sda1/JJug/mag/LIP_JPPNet/output/parsing/128x128/' 

# gt_dir = '/home/lukak/workspace/datasets/lip/val/label/'		            		# path to directory with groundtruth segmentations
gt_dir = '/media/sda1/JJug/mag/LIP_JPPNet/datasets/lip/anotations/parsing_annotations/val_segmentations/'		            		# path to directory with groundtruth segmentations


#CREATING AN ITERABLE LIST OF IMAGE NAMES
images_file = open('/media/sda1/JJug/mag/LIP_JPPNet/datasets/lip/val_id.txt')
images_list = images_file.read().split()
progress = 0


for image in images_list:
	progress+=1
	if(progress%100 == 0):
		print(progress/100,'%')

	#LOAD IMAGES
	try:
		arr_gt = np.array(Image.open(gt_dir+image+'.png').convert('RGB'))
		arr_pred = np.array(Image.open(pred_dir+image+'.png').convert('RGB'))
	except:
		continue
	

	
	#CALCULATE METRICS FOR EACH GARMENT
	for garment in masks_GT:																						 #for single garments

		#SELECT GARMENT COLOR
		#sought_color_RGB = RGB_masks_SCHP[garment]                         #for SCHP
		sought_color_BW = masks_GT[garment]

		#GET BOOLEAN ARRAY OF SELECTED PIXLES (single items)
		pixels_gt = np.all(arr_gt == sought_color_BW, axis=-1) 
		#pixels_pred = np.all(arr_pred == sought_color_RGB, axis=-1)		     #for SCHP
		pixels_pred = np.all(arr_pred == sought_color_BW, axis=-1)        #for JPPNet and openseg 

		#CALCULATE METRICS
		#MISSCLASSIFIED GARMENTS ARE NOT COUNTED (dress vs upperclothes)
		if ((pixels_gt.sum() != 0) and (pixels_pred.sum() != 0)):
			overlap = pixels_pred*pixels_gt
			union = pixels_pred+pixels_gt
			garment_IOU[garment] += overlap.sum()/union.sum()
			garment_count[garment] += 1
			
			precision = overlap.sum()/pixels_pred.sum()
			recall = overlap.sum()/pixels_gt.sum()
			garment_precision[garment] += precision
			garment_recall[garment] += recall
			if (overlap.sum() == 0):
				garment_f1_score[garment] += 0
			else:
				garment_f1_score[garment] += round((2*precision*recall)/(precision+recall), 8)

"""
		#SELECT AREA ON IMAGE
		selected_garment_gt = np.zeros_like(arr_gt)
		selected_garment_pred = np.zeros_like(arr_pred)
		selected_garment_gt[pixels_gt] = [100,1,1]
		selected_garment_pred[pixels_pred] = [1,1,100]
		analisys = Image.fromarray(selected_garment_pred + selected_garment_gt, 'RGB')
		analisys.show()
"""


#FINAL STAGE AFTER ALL THE CALCULATIONS
#PRINT EVERYTHING IN TERMINAL AND A SEPARATE .txt FILE
total_mIOU = 0
file = open("seg_matrics_128x128.txt", "w") 
file.write('\n{0:15}  {1:<10}  {2:<10}  {3:<10}  {4:<10}\n'.format("GARMENT", "IoU", "PRECISION", "RECALL", "F1 SCORE"))
print('{0:15}  {1:<10}  {2:<10}  {3:<10}  {4:<10}\n'.format("GARMENT", "IoU", "PRECISION", "RECALL", "F1 SCORE"))
for garment in masks_GT:
	if garment_count[garment] == 0:
		continue
	garment_IOU[garment] = garment_IOU[garment]/garment_count[garment]
	garment_precision[garment] = garment_precision[garment]/garment_count[garment]
	garment_recall[garment] = garment_recall[garment]/garment_count[garment]
	garment_f1_score[garment] = garment_f1_score[garment]/garment_count[garment]
	file.write('\n{0:15}  {1:<10.4}  {2:<10.4}  {3:<10.4}  {4:<10.4}'.format(garment, garment_IOU[garment], garment_precision[garment], garment_recall[garment], garment_f1_score[garment]))
	print('{0:15}  {1:<10.4}  {2:<10.4}  {3:<10.4}  {4:<10.4}'.format(garment, garment_IOU[garment], garment_precision[garment], garment_recall[garment], garment_f1_score[garment]))
	total_mIOU += garment_IOU[garment]

file.write('\n\nmIoU score for all garments combined: {0:.5}'.format(total_mIOU/len(garment_IOU)))
file.close()
print("\nmIoU score for all garments combined:", total_mIOU/len(garment_IOU))



"""
#SHOWNIG DATA
image = Image.fromarray(selected_garment_gt+selected_garment_pred, 'RGB')
image.show()
print(boolean_pred.sum())
print(boolean_gt.sum())
print(overlap.sum())
print(union.sum())

image_gt = Image.fromarray(selected_garment_gt, 'RGB')
image_pred = Image.fromarray(selected_garment_pred, 'RGB')
image_gt.show()
image_pred.show()
"""



"""
def get_palette(num_cls):
		n = num_cls
		palette = [0] * (n * 3)
		for j in range(0, n):
				lab = j
				palette[j * 3 + 0] = 0
				palette[j * 3 + 1] = 0
				palette[j * 3 + 2] = 0
				i = 0
				while lab:
						palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
						palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
						palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
						i += 1
						lab >>= 3
		return palette

palette = get_palette(20)
for color in range(0, len(palette), 3):
	print(palette[color],palette[color+1],palette[color+2])
"""

"""
for image in os.listdir(images):
	if image.endswith(".png"):
		im = Image.open(image_path + image)
		draw = ImageDraw.Draw(im)
		image_name = image.split(".")[0]
		anno_file = open(pose_annotations+image_name+'.txt')
		anno_list = anno_file.read().split()
		anno_iter = iter(anno_list)
		# another for loop for all the pose annotations
		for anno in anno_iter:
			h = int(anno)
			w = int(next(anno_iter))
			draw.ellipse((h-r, w-r , h+r, w+r), fill=(255, 0, 0), outline=(0, 0, 0))
		
		
		im.save(output_dir+'pose_'+image, quality=100)
		continue
	else:
		print(image+" is not in jpg format!")
		continue
"""