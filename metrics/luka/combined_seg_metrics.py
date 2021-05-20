import os
import numpy as np
from PIL import Image, ImageDraw

combined_RGB_masks_SCHP = {
	"background": [0,0,0],
	"head": [[0,128,0],[0,0,128],[192,0,128],[128,0,0]],
	"scarf": [192,128,0],
	"arms": [[128,128,0],[64,128,128],[192,128,128]],
	"upperclothes": [[128,128,128],[128,0,128],[0,128,128],[64,0,128]],
	"jumpsuits": [64,128,0],
	"pants": [192,0,0],
	"legs": [[0,64,0],[128,64,0]],
	"shoes": [[64,0,0],[0,192,0],[128,192,0]]
} 

combined_masks_GT = {
	"background": [0,0,0],
	"head": [[2,2,2],[4,4,4],[13,13,13],[1,1,1]],
	"scarf": [11,11,11],
	"arms": [[3,3,3],[14,14,14],[15,15,15]],
	"upperclothes": [[7,7,7],[5,5,5],[6,6,6],[12,12,12]],
	"jumpsuits": [10,10,10],
	"pants": [9,9,9],
	"legs": [[16,16,16],[17,17,17]],
	"shoes": [[8,8,8],[18,18,18],[19,19,19]]
} 

garment_IOU = {
	"background": 0,
	"head": 0,
	"scarf": 0,
	"arms": 0,
	"upperclothes": 0,
	"jumpsuits": 0,
	"pants": 0,
	"legs": 0,
	"shoes": 0
} 

garment_precision = {
	"background": 0,
	"head": 0,
	"scarf": 0,
	"arms": 0,
	"upperclothes": 0,
	"jumpsuits": 0,
	"pants": 0,
	"legs": 0,
	"shoes": 0
} 

garment_recall = {
	"background": 0,
	"head": 0,
	"scarf": 0,
	"arms": 0,
	"upperclothes": 0,
	"jumpsuits": 0,
	"pants": 0,
	"legs": 0,
	"shoes": 0
} 

garment_f1_score = {
	"background": 0,
	"head": 0,
	"scarf": 0,
	"arms": 0,
	"upperclothes": 0,
	"jumpsuits": 0,
	"pants": 0,
	"legs": 0,
	"shoes": 0
} 

garment_count = {
	"background": 0,
	"head": 0,
	"scarf": 0,
	"arms": 0,
	"upperclothes": 0,
	"jumpsuits": 0,
	"pants": 0,
	"legs": 0,
	"shoes": 0
} 

#DATA PATHS
pred_dir = '/home/lukak/workspace/datasets/seg_result/lip/hrnet_w48_ocr_1_val/label/'
#pred_dir = '/home/lukak/workspace/datasets/lip/lip_JPPNet_predictions/' 
#pred_dir = '/home/lukak/workspace/datasets/lip/lip_SCHP_predictions/'         # path to directory with predicted segmentations
gt_dir = '/home/lukak/workspace/datasets/lip/val/label/'                    # path to directory with groundtruth segmentations

#CREATING AN ITERABLE LIST OF IMAGE NAMES
images_file = open('/home/lukak/workspace/datasets/lip/lip/val_id.txt')
images_list = images_file.read().split()
progress = 0

for image in images_list:
	progress+=1
	if(progress%100 == 0):
		print(progress/100,'%')

	#LOAD IMAGES
	arr_gt = np.array(Image.open(gt_dir+image+'.png').convert('RGB'))
	arr_pred = np.array(Image.open(pred_dir+image+'.png').convert('RGB'))
	
	#CALCULATE METRICS FOR EACH GARMENT
	for garment in combined_masks_GT:                                    #for combined garments

		pixels_gt = np.full(arr_gt[:,:,1].shape, False)
		pixels_pred = np.full(arr_pred[:,:,1].shape, False)

		#SELECT GARMENT COLOR
		#list_color_RGB = combined_RGB_masks_SCHP[garment]                         #for SCHP
		list_color_BW = combined_masks_GT[garment]

		if(garment == 'background' or garment == 'scarf' or garment == 'jumpsuits' or garment == 'pants'):
			#GET BOOLEAN ARRAY OF SELECTED PIXLES (single item)
			pixels_gt += np.all(arr_gt == list_color_BW, axis=-1) 
			pixels_pred += np.all(arr_pred == list_color_BW, axis=-1)        				#for JPPNet and openseg
			#pixels_pred += np.all(arr_pred == list_color_RGB, axis=-1)         		#for SCHP
		else:
			#GET BOOLEAN ARRAY OF SELECTED PIXLES (combined items)
			for color_BW in list_color_BW:
				pixels_gt += np.all(arr_gt == color_BW, axis=-1) 
				pixels_pred += np.all(arr_pred == color_BW, axis=-1)       					 	#for JPPNet and openseg
			#for sought_color_RGB in list_color_RGB:
			#	pixels_pred += np.all(arr_pred == sought_color_RGB, axis=-1)        	#for SCHP

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
		selected_garment_pred[pixels_pred] = [1,100,100]
		analisys = Image.fromarray(selected_garment_gt+selected_garment_pred, 'RGB')
		analisys.show()
		print(garment)
"""







#FINAL STAGE AFTER ALL THE CALCULATIONS
#PRINT EVERYTHING IN TERMINAL AND A SEPARATE .txt FILE
total_mIOU = 0
file = open("seg_matrics_test_combined.txt", "w") 
file.write('\n{0:15}  {1:<10}  {2:<10}  {3:<10}  {4:<10}\n'.format("GARMENT", "IoU", "PRECISION", "RECALL", "F1 SCORE"))
print('{0:15}  {1:<10}  {2:<10}  {3:<10}  {4:<10}\n'.format("GARMENT", "IoU", "PRECISION", "RECALL", "F1 SCORE"))
for garment in combined_masks_GT:
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