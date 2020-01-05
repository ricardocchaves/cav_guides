import cv2 as cv
import numpy as np

# Video Capture Class. Only supports 4:2:0, 4:2:2 and 4:4:4 chroma subsampling formats
class VideoCaptureYUV:
	def __init__(self, filename):
		self.f = open(filename, 'rb')
		self.header = self.f.readline().decode("utf-8")

		# Reading WIDTH from file
		w_start = self.header.find("W")
		w_end = self.header.find(" ",w_start)
		self.width = int(self.header[w_start+1:w_end])

		# Reading HEIGHT from file
		h_start = self.header.find("H")
		h_end = self.header.find(" ",h_start)
		self.height = int(self.header[h_start+1:h_end])

		# Reading COLOR SPACE from file
		# https://linux.die.net/man/5/yuv4mpeg
		"""
		420jpeg  - 4:2:0 with JPEG/MPEG-1 siting (default)
		422      - 4:2:2, cosited
		444      - 4:4:4 (no subsampling)
		"""
		c_start = self.header.find("C")
		if c_start>0:
			c_end = self.header.find(" ",c_start)
			color = self.header[c_start+1:c_end]
		else:
			color = "420"
		if "444" in color:
			self.color_format = "444"
			self.width_chroma = self.width
			self.height_chroma = self.height
			self.frame_len = self.width * self.height * 3
			self.shape = (self.height, self.width, 3)
		elif "422" in color:
			self.color_format = "422"
			self.width_chroma = self.width / 2
			self.height_chroma = self.height
			self.frame_len = self.width * self.height * 2
			self.shape = (self.height, self.width, 3)
		elif "420" in color:
			self.color_format = "420"
			self.width_chroma = self.width // 2
			self.height_chroma = self.height // 2
			self.frame_len = self.width * self.height * 3 // 2
			self.shape = (self.height, self.width, 3)
		else:
			raise Exception("Incompatible color space!")

		self.f.readline() # Advancing to start of 1st frame

	def read_raw(self):
		raw = self.f.read(self.frame_len)
		if len(raw) != self.frame_len:
			return False, None
		
		cnt = self.width*self.height
		cnt_chroma = self.width_chroma*self.height_chroma
		y = np.frombuffer(raw, dtype=np.uint8, count= cnt)
		u = np.frombuffer(raw, dtype=np.uint8, count=cnt_chroma, offset=cnt)
		v = np.frombuffer(raw, dtype=np.uint8, count=cnt_chroma, offset=cnt+cnt_chroma)
		frame = Frame(y,u,v, self.height, self.height_chroma, self.width, self.width_chroma)
		return True, frame

	@staticmethod
	def split_frame(frame):
		yuv = frame.reshape((frame.shape[0]*frame.shape[1],3))
		y,u,v = yuv.transpose()
		return y,u,v


class Frame:
	def __init__(self, y, u, v, height, height_chroma, width, width_chroma):
		self.y = y
		self.u = u
		self.v = v
		self.height = height
		self.height_chroma = height_chroma
		self.width = width
		self.width_chroma = width_chroma

	def get(self, height, width, color=None):
		if color == None:
			return[self.y[height*self.height + width],self.u[height*self.height_chroma + width],self.v[height*self.height_chroma + width]]
		elif color == 0:
			return self.y[height*self.height + width]
		elif color == 1:
			return self.u[height*self.height_chroma + width]
		elif color == 2:
			return self.v[height*self.height_chroma + width]

	def getMatrix(self, color):
		if color == 0:
			return self.y.reshape((self.height, self.width))
		elif color == 1:
			return self.u.reshape((self.height_chroma, self.width_chroma))
		elif color == 2:
			return self.v.reshape((self.height_chroma, self.width_chroma))
