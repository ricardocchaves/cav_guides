import cv2 as cv
import sys
import numpy as np
from ex_07 import quantize
from math import ceil, floor, log10

# 8)
# Print SNR of image in relation to original
# Print maximum per pixel absolute error
# From doc: https://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html#image-similarity-psnr-and-ssim
def SNR(img,original):
	s = cv.absdiff(img,original) # |img-original|
	s_max = [np.sum(i) for j in s for i in j]
	worstValue = np.max(s_max)
	worstPerChannel = max(np.max(s[:,:,0]),np.max(s[:,:,1]),np.max(s[:,:,2]))
	print("Worst value: {}, Worst per channel: {}".format(worstValue,worstPerChannel))

	cv.imshow('noisy', img)
	cv.imshow('original', original)
	cv.waitKey()
	#print(dir(s1))
	#s1.convertTo(s1,cv.CV_32F) # cannot make a square on 8 bits ??
	s = s*s # |img-original|^2

	s0 = np.sum(s[:,:,0])
	s1 = np.sum(s[:,:,1])
	s2 = np.sum(s[:,:,2])
	
	sse = s0 + s1 + s2
	if(sse <= 1e-10):
		return 0
	else:
		mse = sse/(img.shape[2] * img.shape[0]*img.shape[1] )
		psnr = 10.0*log10((255*255)/mse)
		return psnr

"""
https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
"""
def noisy(noise_typ,image):
	if noise_typ == "gauss":
		row,col,ch= image.shape
		mean = 0
		var = 0.1
		sigma = var**0.5
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = gauss.reshape(row,col,ch)
		noisy = image + gauss
		return noisy
	elif noise_typ == "s&p":
		row,col,ch = image.shape
		s_vs_p = 0.5
		amount = 0.04
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i, int(num_salt))
		for i in image.shape]
		out[tuple(coords)] = 1
		
		# Pepper mode
		num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i, int(num_pepper))
		for i in image.shape]
		out[tuple(coords)] = 0
		return out
	elif noise_typ == "poisson":
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy
	elif noise_typ =="speckle":
		row,col,ch = image.shape
		gauss = np.random.randn(row,col,ch)
		gauss = gauss.reshape(row,col,ch)        
		noisy = image + image * gauss
		return noisy

def main():
	print("USAGE: python3 {} srcFile".format(sys.argv[0]))
	src = sys.argv[1]
	#snr = SNR(noisy('s&p',cv.imread(src)),cv.imread(src))
	snr = SNR(quantize(src,1),cv.imread(src))
	print(snr)

if __name__=="__main__":
	main()