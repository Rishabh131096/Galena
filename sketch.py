import numpy as np
import math
import cv2
import os

#----------------MATRIX INITIALIZATION-----------------
D = np.zeros((10,500)) #Histogram with distance frequency for training
A = np.zeros((10,360)) #Histogram with angle frequency for training
P = np.zeros((10,500)) #Histogram with distance frequency for predicting
B = np.zeros((10,360))
temp = np.zeros((10,360))
ind = 0


shapes = ["Batman", "Bicycle", "Circle", "Hexagon", "Pentagon", "Square", "Triangle"]


#--------FUNCTION TO CALCULATE DISTANCE---------------
def mag(x,y):
	ret = np.float64(x*x + y*y)
	ret = math.sqrt(ret)
	
	return ret


#--------FUNCTION--------------------------
def get_this(x,y,w,h):
	global ind
	center_x = w/2
	center_y = h/2
	
	x = x - center_x
	y = y - center_y
	
	dist = mag(x,y)
	
	angle = math.atan2(y,x)
	if angle < 0:
		angle = angle + 2*math.pi
	angle = math.degrees(angle)
	
	dist = round(dist)
	dist = int(dist)
	angle = round(angle)
	angle = int(angle)
	angle = angle%360
	
	return dist,angle #also return the angle

	
def train():
	global A
	global D
	data_folder_path = "training-data"
	subject_images_names = os.listdir(data_folder_path)
	
	ind = 0
	for image_name in subject_images_names:
		image_path = data_folder_path + "/" + image_name
		#print image_path
		img = cv2.imread(image_path,0)
		ret,thresh = cv2.threshold(img,100,255,0)
		img = thresh
		#cv2.imshow('H',img)
		#cv2.waitKey(1000)
	
		w,h = img.shape
		for x in range(0,w):
			for y in range(0,h):
				if img.item(x,y)==0:
					r = get_this(x,y,w,h)
					D[ind][r[0]] += 1
					temp[ind][r[1]] += 1
					
		sum=0
		num=0
		for i in range(0,360):
			if temp[ind][i]>0:
				sum+=i
				num+=1
		avg = sum/num
		print avg
		for i in range(0,360):
			index = i-avg
			if(index<0):
				index = 360+index
			A[ind][index] = temp[ind][i]
		ind +=1

def predict():
	global P
	
	data_folder_path = "test-data"
	subject_images_names = os.listdir(data_folder_path)
	ind = 0
	for image_name in subject_images_names:
		image_path = data_folder_path + "/" + image_name
		#print image_path
		img = cv2.imread(image_path,0)
		ret,thresh = cv2.threshold(img,90,255,0)
		img = thresh
		#cv2.imshow('H',img)
		#cv2.waitKey(1000)
	
		w,h = img.shape
		for x in range(0,w):
			for y in range(0,h):
				if img.item(x,y)==0:
					r = get_this(x,y,w,h)
					P[ind][r[0]] += 1
					temp[ind][r[1]] += 1
		sum=0
		num=0
		for i in range(0,360):
			if temp[ind][i]>0:
				sum+=i
				num+=1
		avg = sum/num
		print avg
		for i in range(0,360):
			index = i-avg
			if(index<0):
				index = 360+index
			B[ind][index] = temp[ind][i]
		ind +=1
		
def calc_dist():
	global D
	global P
	d=0
	#print D
	#print P
	
	data_folder_path = "test-data"
	test_images_names = os.listdir(data_folder_path)
	data_folder_path = "training-data"
	training_images_names = os.listdir(data_folder_path)
	l = len(test_images_names)
	m = len(training_images_names)
	for i in range(0,l):
		print shapes[i] + ":"
		for j in range(0,m):
			d=0
			a=0
			for k in range(0,360):
				x = A[j][k]-B[i][k]
				a += x*x
			for k in range(0,500):
				x = D[j][k]-P[i][k]
				d += x*x
	
			d = math.sqrt(d)
			a = math.sqrt(a)
			print shapes[j] + " : " + str(d)+ " : " + str(a)
		print "\n"

train()
predict()
calc_dist()