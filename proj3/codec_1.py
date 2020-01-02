import cv2 as cv
import sys
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Value, Manager
import time
from pixel_iteration import pixel_iteration_fast

from VideoCaptureYUV import VideoCaptureYUV

import sys
sys.path.append('..') # Making the other packages in the repository visible.

from proj2.Golomb import Golomb

def main():
	if(len(sys.argv) <= 3):
		print("USAGE: python3 {} inputFile outputFile [args]".format(sys.argv[0]))
		print("Args:")
		print("    -encode")
		print("    -decode")
		return

	inputFile = sys.argv[1]
	outFile = sys.argv[2]
	for arg in sys.argv[3:]:
		if arg == "-encode":
			encode(inputFile,outFile)
		elif arg == "-decode":
			decode(inputFile,outFile)
		
"""
m_y,m_u,m_v = m
c_y = int(math.ceil(math.log(m_y,2)))
div_y = int(math.pow(2,c_y) - m_y)
c_u = int(math.ceil(math.log(m_u,2)))
div_u = int(math.pow(2,c_u) - m_u)
c_v = int(math.ceil(math.log(m_v,2)))
div_v = int(math.pow(2,c_v) - m_v)
"""

def encode(inputFile, outputFile, m=5):
	cap = VideoCaptureYUV(inputFile)

	f = open(outputFile, 'wb')
	header = cap.header.encode('utf-8')
	f.write(header)

	cnt = 1
	current = Value('i',0)
	#manager = multiprocessing.Manager()
    #return_dict = manager.dict()
	pbar = tqdm(total=500)
	while True:
		ret, frame = cap.read_raw()
		if ret:
			"""
			outFrame = np.zeros(cap.shape)
			for h in tqdm(range(cap.height)):
				for w in range(cap.width):
					y,u,v = frame[h,w]
					y_p = y
					u_p = u
					v_p = v

					#y_g = Golomb.to_golomb(y_p,m_y,c_y,div_y)
					#u_g = Golomb.to_golomb(u_p,m_u,c_u,div_u)
					#v_g = Golomb.to_golomb(v_p,m_v,c_v,div_v)

					#outFrame[h,w] = [y_g, u_g, v_g]
					outFrame[h,w] = [y_p,u_p,v_p]
			print("Done")
			writeFrame(f,outFrame,m)
			"""
			
			while current.value == 4:
				time.sleep(0.1) # because 'wait' doesn't exist
			p = Process(target=processFrame,args=(cap.height,cap.width,frame,f,m,cnt,current))
			p.start()
			cnt += 1
			pbar.update(1)
			current.value += 1
		else:
			break
	# After all 500 frames:
	# Find best 'm' for every value in each overall
	# np.array(500*size(overall)*3)
	# [val,val,val,val,val,val,...] use frame_len to separate frames
	# for valueToEncode in np.array:
	#	outputBuffer = Golomb.toGolomb(value)
	# Write metadata and encoded data to output file
	f.close()

def processFrame(height,width,frame,handler,m,ID,current):
	# for pixel:
	# 	Predicted value
	# 	pixel value
	#	value to encode
	#	y_overall += valueToEncode.y
	#	u_overall += valueToEncode.u
	#	v_overall += valueToEncode.v
	s = time.time()
	#outFrame = pixel_iteration_slow(height,width,frame)
	outFrame = np.array(pixel_iteration_fast(height,width,frame))
	writeFrame(handler,outFrame,m)
	#print(time.time()-s)
	current.value -= 1
	# Add y,u,v overall to globalOverall (y,u and v) dict/list in the frame position

def pixel_iteration_slow(height,width,frame):
	outFrame = np.copy(frame)
	for h in range(height):
		for w in range(width):
			y,u,v = frame[h,w]
			y_p = y
			u_p = u
			v_p = v

			outFrame[h,w] = [y_p,u_p,v_p]
	return outFrame

def writeFrame(file_handler, outFrame, m):
	file_handler.write(b'FRAME\n')
	y,u,v = VideoCaptureYUV.split_frame(outFrame)
	outFrame_bytes = y.tobytes()+u.tobytes()+v.tobytes()

	#c = int(math.ceil(math.log(m,2)))
	#div = int(math.pow(2,c) - m)
	#golomb_bytes = Golomb.to_golomb()
	
	file_handler.write(outFrame_bytes)

# c | b
# a | X
def nonlinearPredictor(a,b,c):
	if c >= max(a,b):
		return min(a,b)
	elif c <= min(a,b):
		return max(a,b)
	else:
		return a+b-c

def decode(inputFile, outputFile):
	return 0

if __name__ == "__main__":
	main()