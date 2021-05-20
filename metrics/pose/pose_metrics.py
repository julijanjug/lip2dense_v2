import os
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import distance
from statistics import mean 


#DATA PATHS
pred_dir = '/media/sda1/JJug/mag/LIP_JPPNet/output/pose/val/'							            		# path to directory with predicted segmentations
# pred_dir = '/media/sda1/JJug/mag/LIP_JPPNet/output/pose/
gt_file = '/media/sda1/JJug/mag/LIP_JPPNet/datasets/lip/lip_val_set.csv'

#CREATING AN ITERABLE LIST OF IMAGE NAMES with poses
images_file = open(gt_file)
images_list = images_file.read().split()
progress = 0

distances = dict()
for j in range(16):
		distances[j] = []
	
for line in images_list:
	lineList = line.split(',')  #ground truth
	image = lineList[0].split('.')[0]

	progress+=1
	if(progress%100 == 0):
		print(progress/100,'%')

	#LOAD Prediction
	try:
		predList = open(pred_dir+image+'.txt', 'r').read().split(' ')
	except:
		print('[!] Slika '+pred_dir+image+'.txt'+' ni najdena')
		continue
	
	#CALCULATE METRICS FOR EACH GARMENT
	i=0
	for point_index in range(0,len(predList)-1,2):
		try:
			x = int(predList[point_index])
			y = int(predList[point_index+1])

			gt_x = int(lineList[point_index+1+i])
			gt_y = int(lineList[point_index+2+i])
			i = i+1

			pred = (x,y)
			gt = (gt_x,gt_y)
			dst = distance.euclidean(pred, gt)
			distances[point_index/2].append(dst)
		except:
			# print('[!] Neveljavni podatki za sliko: '+pred_dir+image+'.txt')
			continue





#FINAL STAGE AFTER ALL THE CALCULATIONS
#PRINT EVERYTHING IN TERMINAL AND A SEPARATE .txt FILE
distSumList = []
file = open("pose_matrics_test.txt", "w") 
file.write('\n{0:15}  {1:<10}  {2:<10}  {3:<10}  {4:<10}\n'.format("KEYPOINT", "AVG.DISTANCE", "PRECISION", "RECALL", "F1 SCORE"))
print('{0:15}  {1:<10}  {2:<10}  {3:<10}  {4:<10}\n'.format("KEYPONIT", "AVG.DISTANCE", "PRECISION", "RECALL", "F1 SCORE"))
for i in range(16):
	avgDist = mean(distances[i])
	distSumList = distSumList + distances[i]
	file.write('\n{0:15}  {1:<10.4}  {2:<10.4}  {3:<10.4}  {4:<10.4}'.format(str(i), avgDist, 0.0, 0.0, 0.0))
	print('{0:15}  {1:<10.4}  {2:<10.4}  {3:<10.4}  {4:<10.4}'.format(str(i), avgDist, 0.0, 0.0, 0.0))
	

file.write('\n\nCombined mean distance: {0:.5}'.format(mean(distSumList)))
file.close()
print("\nCombined mean distance:", mean(distSumList))

