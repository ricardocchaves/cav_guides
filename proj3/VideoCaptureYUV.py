import cv2 as cv
import numpy as np

class VideoCaptureYUV:
	def __init__(self, filename):
		self.f = open(filename, 'rb')
		self.header = self.f.readline().decode("utf-8")

		# Reading COLOR SPACE from filename (not always inside file)
		if "444" in filename:
			self.color_format = "444"
		elif "422" in filename:
			self.color_format = "422"
		elif "420" in filename:
			self.color_format = "420"
		else:
			raise Exception("Couldn't find color space! Must be in file name")
		
		# Reading WIDTH from file
		w_start = self.header.find("W")
		w_end = self.header.find(" ",w_start)
		self.width = int(self.header[w_start+1:w_end])

		# Reading HEIGHT from file
		h_start = self.header.find("H")
		h_end = self.header.find(" ",h_start)
		self.height = int(self.header[h_start+1:h_end])

		if self.color_format == "420":
			self.frame_len = self.width * self.height * 3 // 2
			self.shape = (int(self.height*3/2), self.width)
		elif self.color_format == "422":
			self.frame_len = self.width * self.height * 2
			self.shape = (self.height, self.width, 2)
		else: #444
			self.frame_len = self.width * self.height * 3
			self.shape = (self.height, self.width, 3)

		self.f.readline() # Advancing to start of 1st frame

	def read_raw(self):
		raw = self.f.read(self.frame_len)
		if len(raw) != self.frame_len:
			return False, None
		#cnt = len(raw)//3
		#cnt = self.width*self.height
		prod = 1
		for v in self.shape:
			prod *= v
		cnt = prod//3
		y = np.frombuffer(raw, dtype=np.uint8, count=cnt)
		u = np.frombuffer(raw, dtype=np.uint8, count=cnt, offset=cnt)
		v = np.frombuffer(raw, dtype=np.uint8, count=cnt, offset=cnt*2)
		yuv = np.dstack([y,u,v])[0]
		yuv = yuv.reshape(self.shape)
		return True, yuv

	def read(self):
		ret, yuv = self.read_raw()
		if self.color_format == "420":
			bgr = cv.cvtColor(yuv, cv.COLOR_YUV2BGR_NV12)
		elif self.color_format == "422":
			bgr = cv.cvtColor(yuv, cv.COLOR_YUV2BGR_NV21)
		else: #444
			bgr = cv.cvtColor(yuv, cv.COLOR_YUV2BGR_NV21)
		return ret, bgr

	@staticmethod
	def split_frame(frame):
		yuv = frame.reshape((frame.shape[0]*frame.shape[1],3))
		y,u,v = yuv.transpose()
		return y,u,v
