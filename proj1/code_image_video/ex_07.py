import cv2 as cv
import sys
import numpy as np
from math import ceil, floor, log10

# 7)
# Reduce number of bits used to represent each pixel
def quantize(file,bits=8,showDelay=0,fromFile=True):
	if fromFile:
		img = cv.imread(file)
	else:
		img = file

	h = img.shape[0]
	w = img.shape[1]

	maxValue = 2**bits

	quantizedImg = np.zeros(shape=img.shape, dtype=np.uint8)

	for y in range(0,h):
		for x in range(0,w):
			b = floor((img[y,x][0]/256)*maxValue)*(256/maxValue)
			g = floor((img[y,x][1]/256)*maxValue)*(256/maxValue)
			r = floor((img[y,x][2]/256)*maxValue)*(256/maxValue)
			quantizedImg[y,x] = [b,g,r]
	
	cv.imshow('Reduced - {} bits'.format(bits),quantizedImg)
	cv.waitKey(showDelay)
	return quantizedImg

def videoQuantize(file,bits=8):
	# Video
	cap = cv.VideoCapture(file)
	if(cap.isOpened() == False):
		print("Error opening video.")
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret:
			#cv.imshow(file,frame)
			quantize(frame,bits,25,False)

			key = cv.waitKey(25)
			if key==27 or key==113: #ESC or Q
				break
		else:
			break
	cap.release()
	cv.destroyAllWindows()

def main():
	print("USAGE: python3 {} srcFile [OPTIONS]".format(sys.argv[0]))
	print("OPTIONS: --video - use if file is video")
	src = sys.argv[1]
	if(len(sys.argv)>2):
		if(sys.argv[2]=="--video"):
			videoQuantize(src,2)
	else:
		quantize(src,2)

if __name__=="__main__":
	main()