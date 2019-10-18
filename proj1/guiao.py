import cv2 as cv
import os
import sys
import numpy as np
from math import ceil, floor, log10

# 2)
# Copy an image, pixel by pixel
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

# 3)
# Show an image or video
def show(file):
	img = cv.imread(file)
	isImage = True
	try:
		isImage = img.any()
	except:
		isImage = False

	if isImage:
		# Image
		cv.namedWindow(file,cv.WINDOW_NORMAL)
		cv.imshow(file,img)
		#cv.waitKey()
	else:
		# Video
		cap = cv.VideoCapture(file)
		if(cap.isOpened() == False):
			print("Error opening video.")
		while(cap.isOpened()):
			ret, frame = cap.read()
			if ret:
				cv.imshow(file,frame)

				key = cv.waitKey(25)
				if key==27 or key==113: #ESC or Q
					break
			else:
				break
		cap.release()
		cv.destroyAllWindows()

# 5)
# Histogram of an image file
def histogram(img):
	h = img.shape[0]
	w = img.shape[1]
	
	hist = np.zeros(256)

	for y in range(0,h):
		for x in range(0,w):
			val = img[y,x]
			hist[val]+=1
	return hist

def fullHistogram(file,showDelay=0,fromFile=True):
	if fromFile:
		img = cv.imread(file)
	else:
		img = file

	b = histogram(img[:,:,0])
	g = histogram(img[:,:,1])
	r = histogram(img[:,:,2])
	gray = histogram(cv.cvtColor(img,cv.COLOR_BGR2GRAY))

	bar_width = 5 # 5 pixels each bar
	vertical_scale = 16

	width = 256*bar_width
	height = int(max([max(b),max(g),max(r),max(gray)])//vertical_scale)

	#print(width,height)
	histImage = 255*np.ones(shape=(height,width,3), dtype=np.uint8)

	for y in range(0,height):
		for x in range(0,width):
			bgrg = [(b[x//bar_width]//vertical_scale,'b'),
					(g[x//bar_width]//vertical_scale,'g'),
					(r[x//bar_width]//vertical_scale,'r'),
					(gray[x//bar_width]//vertical_scale,'gray'),
					(y,'y')]
			bgrg.sort()
			yPos = bgrg.index((y,'y'))
			if (yPos-1)>=0:
				color = bgrg[yPos-1][1]
				if color=='b':
					histImage[y,x] = (255,0,0)
				elif color=='g':
					histImage[y,x] = (0,255,0)
				elif color=='r':
					histImage[y,x] = (0,0,255)
				elif color=='gray':
					histImage[y,x] = (90,90,90)

	cv.imshow('Histogram',histImage)
	cv.waitKey(showDelay)

def videoHistogram(file):
	# Video
	cap = cv.VideoCapture(file)
	if(cap.isOpened() == False):
		print("Error opening video.")
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret:
			cv.imshow(file,frame)
			fullHistogram(frame,25,False)

			key = cv.waitKey(25)
			if key==27 or key==113: #ESC or Q
				break
		else:
			break
	cap.release()
	cv.destroyAllWindows()

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
	#cv.waitKey(showDelay)
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

# 11)
# Calculate the entropy of image, based on a finite-context model
def entropyThread(img,pixel_val,ret_count,ret_entropy):
	count = 256*[0]
	h = img.shape[0]
	w = img.shape[1]

	# For each pixel, register how many times a value appears after i
	# Look to check if current pixel is after i
	for y in range(0,h):
		for x in range(0,w):
			val = img[y,x]
			if (x-1)>=0:
				if img[y,x-1] == pixel_val:
					count[val] += 1
			else:
				if (y-1)>=0:
					if img[y-1,w-1] == pixel_val:
						count[val] += 1

	if np.sum(count)>0:
		hist = count/np.sum(count)
	else:
		hist = count
	hist = list(filter(lambda p: p > 0, np.ravel(hist))) # Remove probabilities equal to 0
	entropy = -np.sum(np.multiply(hist, np.log2(hist)))
	print("Pixel val {} entropy: {}".format(pixel_val,entropy))
	ret_count[pixel_val] = count
	ret_entropy[pixel_val] = entropy

def getEntropy(file,order=0):
	img = cv.imread(file)
	img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

	histOriginal = histogram(img)
	if order == 0:
		hist = histOriginal/img.size
		hist = list(filter(lambda p: p > 0, np.ravel(hist))) # Remove probabilities equal to 0
		entropy = -np.sum(np.multiply(hist, np.log2(hist)))
		return entropy
	elif order == 1:

		import threading
		ret_count = 256*[0]
		ret_entropy = 256*[0]

		class MyThread(threading.Thread):
			def __init__(self, i, img, ret_count, ret_entropy):
				threading.Thread.__init__(self)
				self.threadID = i
				self.img = img
				self.i = i
				self.ret_count = ret_count
				self.ret_entropy = ret_entropy
			def run(self):
				entropyThread(self.img,self.i,self.ret_count,self.ret_entropy)

		threads = 256*[0]
		for i in range(0,256):
			threads[i] = MyThread(i,img,ret_count,ret_entropy).start()
		
		for t in threads:
			if t:
				t.join()

		print("Done! Calculating global entropy...")
		global_entropy = np.sum(ret_entropy)/len(ret_entropy)
		print("Entropy: {}".format(global_entropy))

# Calculate entropy Order 1 of img
def getEntropy_nonThread(file):
	from tqdm import tqdm

	img = cv.imread(file)
	img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

	h = img.shape[0]
	w = img.shape[1]

	m = {}
	print("Initializing data structure...")
	for i in range(0,256):
		m.update({i:256*[0]}) # pixel_val: [count,count,count,...]

	# For each pixel, register how many times a value appears after i
	# Look to check if current pixel is after i
	print("Searching pixel values...")
	for y in tqdm(range(0,h)):
		for x in range(0,w):
			val = img[y,x]
			for pixel_val in range(0,256):
				count = m[i]
				if (x-1)>=0:
					if img[y,x-1] == pixel_val:
						count[val] += 1
				else:
					if (y-1)>=0:
						if img[y-1,w-1] == pixel_val:
							count[val] += 1
				m[i] = count

	entropies = []
	print("Calculating entropies...")
	np.seterr(divide='ignore', invalid='ignore')
	for i in tqdm(range(0,256)):
		# Calculate entropy of each value
		count = m[i]
		count = count/np.sum(count)
		count = list(filter(lambda p: p > 0, np.ravel(count))) # Remove probabilities equal to 0
		entropy = -np.sum(np.multiply(count, np.log2(count)))
		entropies.append(entropy)

	print("Overall entropy: {}".format(np.sum(entropies)/len(entropies)))

	entropies = list(filter(lambda p: p > 0, np.ravel(entropies)))
	print("Overall entropy (excluding 0): {}".format(np.sum(entropies)/len(entropies)))

def main():
	"""
	if len(sys.argv) != 3:
		print("USAGE: {} srcImage destImage".format(sys.argv[0]))
		return -1
	"""

	src = sys.argv[1]
	#dst = sys.argv[2]
	#copy(src,dst)
	#show(src)
	#print(histogram(cv.imread(src)))
	#fullHistogram(src)
	#videoHistogram(src)
	#quantize(src,1)
	#videoQuantize(src,1)
	#snr = SNR(noisy('s&p',cv.imread(src)),cv.imread(src))
	#snr = SNR(quantize(src,1),cv.imread(src))
	#print(snr)
	#print(getEntropy(src,1))
	getEntropy_nonThread(src)
	#cv.waitKey(4000000)

if __name__=="__main__":
	main()