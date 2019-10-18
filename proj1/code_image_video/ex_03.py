import cv2 as cv
import sys

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
		cv.waitKey()
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

def main():
	print("USAGE: python3 {} srcFile".format(sys.argv[0]))
	src = sys.argv[1]
	show(src)

if __name__=="__main__":
	main()