import cv2 as cv
import sys
import math

def main():
	if (len(sys.argv) < 2):
		print("USAGE: python3 {} videoFile".format(sys.argv[0]))
		return

	fname = sys.argv[1]
	cap = cv.VideoCapture(fname)
	framerate = math.ceil(cap.get(cv.CAP_PROP_FPS))
	
	if(not cap.isOpened()):
		print("Error opening video!")
		return

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret:
			cv.imshow('Frame',frame)
			if cv.waitKey(int((1/framerate)*1000)) & 0xFF == ord('q'):
				break
		else:
			break

if __name__ == "__main__":
	main()
