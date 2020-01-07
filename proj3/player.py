import cv2 as cv
import sys
import math
from VideoCaptureYUV import VideoCaptureYUV
import time

def main():
	if (len(sys.argv) < 2):
		print("USAGE: python3 {} videoFile".format(sys.argv[0]))
		return

	fname = sys.argv[1]
	cap = VideoCaptureYUV(fname)
	framerate = 50

	s = time.time()
	while True:
		ret, frame = cap.read()
		if ret:
			#for ch in range(3):
				#print(len(frame))
			cv.imshow('Frame',frame)
			if cv.waitKey() & 0xFF == ord('q'):
				break
			#if cv.waitKey(int((1/framerate)*1000)) & 0xFF == ord('q'):
				#break
		else:
			break
	print("Length: {:.2f}s".format(time.time()-s))

if __name__ == "__main__":
	main()
