import cv2
import numpy as np
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def gray_scale(img):
	#display the gray scale imag
	np.dot(img[...,:3], [0.299, 0.587, 0.114])
	cv2.imshow(img)
	return img

def main():
        #Open the image

        img = mpimg.imread(sys.argv[1])
        gray = gray_scale(img)

if __name__ == "__main__": main()
