import cv2 as cv
import numpy as np
import sys

def copy(srcFile, dstFile):
	img = cv.imread(srcFile)
	h = img.shape[0]
	w = img.shape[1]

	copyImg = np.zeros(shape=img.shape, dtype=np.uint8)

	for y in range(0,h):
		for x in range(0,w):
			copyImg[y,x] = img[y,x]
	
	cv.imwrite(dstFile,copyImg)
	return 0

def main():
	print("USAGE: python3 {} srcFile dstFile".format(sys.argv[0]))
	src = sys.argv[1]
	dst = sys.argv[2]
	copy(src,dst)

if __name__ == "__main__":
	main()