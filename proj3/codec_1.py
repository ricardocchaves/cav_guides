import cv2 as cv
import sys
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Value, Manager, cpu_count
import time
import warnings

from pixel_iteration import pixel_iteration_fast

from VideoCaptureYUV import VideoCaptureYUV

import sys
sys.path.append('..') # Making the other packages in the repository visible.

from proj2.Golomb import Golomb
import proj2.encode as GolombEst

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

def encode(inputFile, outputFile):
	cap = VideoCaptureYUV(inputFile)

	header = cap.header.encode('utf-8')

	cnt = 1
	manager = Manager()
	shared_dict = manager.dict()
	currentProcess = 0
	processes = {}
	pbar = tqdm(total=500)
	while True:
		ret, frame = cap.read_raw()
		if ret:
			if len(processes) == cpu_count():
			#if len(processes) == 1:
				for p in processes:
					processes[p].join()
					processes[p].terminate()
				processes.clear()
				currentProcess = 0
				shared_dict_size = len(shared_dict)*3*len(shared_dict[1][0])//(1024*1024)

				print("\033[A\033[AShared dict. {} frames, {} MB ".format(len(shared_dict),shared_dict_size))
				#pbar.update(cpu_count())
				time.sleep(0.2)

			p = Process(target=processFrame,args=(cap.height,cap.width,frame,cnt,shared_dict))
			p.start()
			pbar.update(1)
			
			processes[currentProcess] = p
			cnt += 1
			currentProcess += 1
		else:
			break

	y_overall = []
	u_overall = []
	v_overall = []
	for frame in shared_dict:
		y,u,v = shared_dict[frame]
		y_overall.append(y)
		u_overall.append(u)
		v_overall.append(v)

	shared_dict.clear()

	values = (y_overall,u_overall,v_overall)

	#### Find best 'm' for every value in each overall
	# Calculate probabilty of each sample value after predictor is applied
	sample_probability_y = GolombEst.prob([r for (l,r) in y_overall])
	sample_probability_u = GolombEst.prob([r for (l,r) in u_overall])
	sample_probability_v = GolombEst.prob([r for (l,r) in v_overall])
	
	# Map symbols to sample values according to probability
	#{ symbol: (value, probability)}
	y_symbolToValue, y_valueToSymbol  = GolombEst.map_symbols(sample_probability_y)
	u_symbolToValue, u_valueToSymbol  = GolombEst.map_symbols(sample_probability_u)
	v_symbolToValue, v_valueToSymbol  = GolombEst.map_symbols(sample_probability_v)

	# Find best Golomb Parameters
	alpha_y, m_y = GolombEst.findBestGolomb(y_symbolToValue,False)
	alpha_u, m_u = GolombEst.findBestGolomb(u_symbolToValue,False)
	alpha_v, m_v = GolombEst.findBestGolomb(v_symbolToValue,False)

	symbolsY = {}
	symbolsU = {}
	symbolsV = {}
	# Removing probabilities from symbolToValue map. valueToSymbol already doesn't have them.
	for k in y_symbolToValue:
		symbolsY.update({k:y_symbolToValue[k][0]})
	for k in u_symbolToValue:
		symbolsU.update({k:u_symbolToValue[k][0]})
	for k in v_symbolToValue:
		symbolsV.update({k:v_symbolToValue[k][0]})

	sym = []
	sym.append(symbolsY, y_valueToSymbol)
	sym.append(symbolsU, u_valueToSymbol)
	sym.append(symbolsV, v_valueToSymbol)

	m = []
	m.append(m_y)
	m.append(m_u)
	m.append(m_v)

	encodeValues(outputFile, header, values, sym, m)

def encodeValues(fname,header,values, sym, m_list):
	f = open(fname, 'wb')
	f.write(header)
	
	# Writing metadata
	metadata = {}
	for ch in range(len(sym)):
		symbolToValue,_ = sym[ch]
		metadata[ch] = symbolToValue
		
	from pickle import dumps
	metadata = dumps(metadata) # bytes
	f.write(metadata)

	f.write("\nData\n")

	for ch in range(len(values)):
		_,valueToSymbol = sym[ch]
		vals = values[ch]
		m = m_list[ch]
		c = int(math.ceil(math.log(m,2)))
		div = int(math.pow(2,c) - m)
		golomb_result = b""
		for val in vals:
			val = valueToSymbol[val]
			golomb_result += Golomb.to_golomb(val, m, c, div)
		

def processFrame(height,width,frame,ID,shared_dict):
	#y_overall, u_overall, v_overall = pixel_iteration_slow(height,width,frame)
	y_overall,u_overall,v_overall = pixel_iteration_fast(height,width,frame)
	# Add y,u,v overall to globalOverall (y,u and v) dict/list in the frame position
	shared_dict[ID] = (y_overall,u_overall,v_overall)

def pixel_iteration_slow(height,width,frame):
	overalls = ([],[],[]) # y,u,v
	for h in range(height):
		for w in range(width):
			y,u,v = frame[h,w]
			y_p, u_p, v_p = predictor(frame,h,w)

			diff_y = y - y_p
			diff_u = u - u_p
			diff_v = v - v_p

			overalls[0].append(diff_y)
			overalls[1].append(diff_u)
			overalls[2].append(diff_v)
	return overalls

def writeFrame(file_handler, outFrame, m):
	file_handler.write(b'FRAME\n')
	y,u,v = VideoCaptureYUV.split_frame(outFrame)
	outFrame_bytes = y.tobytes()+u.tobytes()+v.tobytes()

	
	file_handler.write(outFrame_bytes)

def predictor(frame,pos_h,pos_w):
	if pos_w-1>=0:
		a = frame[pos_h,pos_w-1]
	else:
		a = (0,0,0)

	if pos_h-1>=0:
		b = frame[pos_h-1,pos_w]
	else:
		b = (0,0,0)

	if pos_h-1>=0 and pos_w-1>=0:
		c = frame[pos_h-1,pos_w-1]
	else:
		c = (0,0,0)
	
	y_p = _nonLinearPredictor(int(a[0]),int(b[0]),int(c[0]))
	u_p = _nonLinearPredictor(int(a[1]),int(b[1]),int(c[1]))
	v_p = _nonLinearPredictor(int(a[2]),int(b[2]),int(c[2]))
	return y_p,u_p,v_p

# c | b
# a | X
def _nonLinearPredictor(a,b,c):
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
