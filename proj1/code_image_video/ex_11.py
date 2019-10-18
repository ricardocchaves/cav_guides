import cv2 as cv
import sys
import numpy as np

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
	print("USAGE: python3 {} srcFile".format(sys.argv[0]))
	src = sys.argv[1]
	getEntropy_nonThread(src)

if __name__=="__main__":
	main()