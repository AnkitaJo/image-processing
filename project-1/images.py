#importing the necessary libraries

import numpy as np
import cv2
import copy
#from PIL import Image
import sys


#Question 1
#loading a color image in gray-scale
#img = cv2.imread('../../Desktop/House_sparrow04.jpg',1)
def read_image(img):
	image = cv2.imread(img)
	oimg = cv2.resize(image, (256, 256))
	return oimg

def gray_scale(img):
	#display the gray scale image
	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	print(grey.shape)
	cv2.imwrite('gray.jpg', grey)
	image = cv2.imread("gray.jpg")
	oimg = cv2.resize(image, (256, 256))
	print(oimg.shape)
	return oimg


def noiseGaussian(img):

	row,col,ch= img.shape
	mean = 0
	var = 100
	sigma = var**0.5
	gauss = np.random.normal(mean,sigma,(row,col,ch))
	gauss = gauss.reshape(row,col,ch)
	gauss = gauss.astype(int)
	noisy = img + gauss
	cv2.imwrite('nois.jpg', noisy)
	nn = cv2.imread("nois.jpg")
	"""
	n = noisy.astype(int)
	for i in n:
			for j in i:
				for k in j:
					if (k < 0) or (k > 255):
						print(str(j) + " ")
	#np.savetxt("res.txt", n)
	"""
	return nn

def noiseSaltPepper(img):
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
	return out

def smoothening(img):
	kernel = np.ones((9,9),np.float32)/81
	dst = cv2.filter2D(img,-1,kernel)
	return(dst)

def medianFilter(img):
	median = cv2.medianBlur(img,5)
	return median


def thresholding(img):
	#o2 = gray_scale(img)
	#ret,thresh1 = cv2.threshold(o2,127,255,cv2.THRESH_BINARY)
	return img

"""
def ithresholding(img):
	o2 = gray_scale(img)
"""

"""

def labelComponents(img):
	o2 = gray_scale(img)
	ret,thresh1 = cv2.threshold(o2,127,255,cv2.THRESH_BINARY)
	ret1, labels = cv2.connectedComponents(thresh1)

	label_hue = np.uint8(179*labels/np.max(labels))
	blank_ch = 255*np.ones_like(label_hue)
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
	labeled_img[label_hue==0] = 0
	return labeled_img
"""



def display1(img1):
	cv2.imshow('Result', img1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def display(img1,img2):
	

	#display1(img1)
	#display1(img2)


	numpy_horizontal = np.hstack((img1, img2))
	#print(numpy_horizontal.shape)
	#numpy_horizontal_concat = np.concatenate((img1, img2), axis=1)
	cv2.imshow('Result', numpy_horizontal)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


	#display1(img1)
	#display1(img2)
	




	
	



	

"""

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






def main():
	#Open the image

	o1 = read_image(sys.argv[1])
	sample = copy.deepcopy(o1)
	o2 = gray_scale(sample)	
	o3 = noiseGaussian(sample)
	o4 = noiseSaltPepper(sample)
	o5 = smoothening(sample)
	o6 = medianFilter(sample)
	o7 = thresholding(sample)
	#o8 = labelComponents(sample)
	#print(o2.shape)
	
	#display(o1,o2)



if __name__ == "__main__": main()
