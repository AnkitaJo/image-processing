#importing the necessary libraries

import numpy as np
import cv2
import Image, ImageDraw
import sys


#Question 1
#loading a color image in gray-scale
#img = cv2.imread('../../Desktop/House_sparrow04.jpg',1)
def read_image(img):
	oimg = cv2.imread(img)
	#oimg = cv2.resize(image, (0, 0), None, .25, .25)
	return oimg

def gray_scale(img):
	#display the gray scale image
	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	return grey

def display(img1,img2):

	
	image1 = cv2.resize(img1, (0, 0), None, .25, .25)
	image2 = cv2.resize(img2, (0, 0), None, .25, .25)
	numpy_horizontal = np.hstack((image1, image2))
	numpy_horizontal_concat = np.concatenate((image1, image2), axis=1)
	cv2.imshow('Result', numpy_horizontal_concat)
	
	

#Question 2
#corrupting the image by noise
#1. Gaussian noise

"""
row,col,ch= img.shape
mean = 0
var = 0.05
sigma = var**0.5
gauss = np.random.normal(mean,sigma,(row,col,ch))
gauss = gauss.reshape(row,col,ch)
noisy = img + gauss
cv2.imshow('image',noisy)
cv2.waitKey(0)
cv2.destroyAllWindows()

#2. Salt-pepper noise introduction
row,col,ch = img.shape
s_vs_p = 0.5
amount = 0.004
out = np.copy(img)
# Salt mode
num_salt = np.ceil(amount * img.size * s_vs_p)
coords = [np.random.randint(0, i - 1, int(num_salt))
	for i in img.shape]
out[coords] = 1

# Pepper mode
num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
coords = [np.random.randint(0, i - 1, int(num_pepper))
	for i in img.shape]
out[coords] = 0
cv2.imshow('image',out)
cv2.waitKey(0)
cv2.destroyAllWindows()

#3. Poisson noise introduction
vals = len(np.unique(img))
vals = 2 ** np.ceil(np.log2(vals))
noisy2 = np.random.poisson(img * vals) / float(vals)
cv2.imshow('image',noisy2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#4. Speckle noise introduction
row,col,ch = img.shape
gauss = np.random.randn(row,col,ch)
gauss = gauss.reshape(row,col,ch)        
noisy3 = img + img * gauss
cv2.imshow('image',noisy3)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
#Question 3
#Smoothening an image using averaging technique
img = cv2.imread('../../Desktop/House_sparrow04.jpg',1)
kernel = np.ones((9,9),np.float32)/81
dst = cv2.filter2D(img,-1,kernel)
cv2.imshow('image',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
#Question 4
#Applying Median filtering to an image
img = cv2.imread('../../Desktop/House_sparrow04.jpg',1)
median = cv2.medianBlur(img,5)
cv2.imshow('image1',median)
cv2.imshow('image2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
#Question 5
#Converting a picture to binary using simple thresholding technique

#image should be a gray scale image to use this technique
img = cv2.imread('../../Desktop/House_sparrow04.jpg',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow('image',thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Question 6
#Convert a picture to binary using the P-Tile Method
#def pTile(img,x):




#Question 7
#Converting a picture to binary using the Iterative Thresholding method

#Question 8
#Label Compponents of a black and white image
def labelComponents(img):
	#convert the image in binary
	#


#Question 10
#Other functions: Histogram
"""
def main():
	#Open the image

	o1 = read_image(sys.argv[1])
	o2 = gray_scale(o1)
	display(o1,o2)


if __name__ == "__main__": main()
