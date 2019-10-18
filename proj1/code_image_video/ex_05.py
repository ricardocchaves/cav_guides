import cv2 as cv
import sys
import numpy as np

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

def main():
	print("USAGE: python3 {} srcFile [OPTIONS]".format(sys.argv[0]))
	print("OPTIONS: --video - use if file is video")
	src = sys.argv[1]
	if(len(sys.argv)>2):
		if(sys.argv[2]=="--video"):
			videoHistogram(src)
	else:
		fullHistogram(src)
	

if __name__=="__main__":
	main()