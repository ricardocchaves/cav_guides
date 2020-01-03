import cv2 as cv
import sys
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Value, Manager, cpu_count
import time
import warnings

from pixel_iteration import pixel_iteration_fast
from pixel_iteration import toGolomb

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

	output = open(outputFile, 'wb')
	output.write(header)

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

				dumpFrames(shared_dict,output)

				shared_dict_size = len(shared_dict)*3*cap.width*cap.height//(1024*1024)

				print("\033[A\033[A Shared dict: {} frames, {} MB ".format(len(shared_dict),shared_dict_size))
				time.sleep(0.2)

			p = Process(target=processFrame,args=(cap.height,cap.width,frame,cnt,shared_dict))
			p.start()
			pbar.update(1)
			
			processes[currentProcess] = p
			cnt += 1
			currentProcess += 1
		else:
			break
	dumpFrames(shared_dict,output,0)

def encodeValues(fhandler,values, sym, m_list):
	# Writing metadata
	metadata = {}
	for ch in range(len(sym)):
		symbolToValue,_ = sym[ch]
		metadata[ch] = symbolToValue
		
	from pickle import dumps
	metadata = dumps(metadata) # bytes
	fhandler.write(metadata)

	fhandler.write(b"\nDATA\n")

	for ch in range(len(values)):
		_,valueToSymbol = sym[ch]
		vals = values[ch]
		m = int(m_list[ch])
		c = int(math.ceil(math.log(m,2)))
		div = int(math.pow(2,c) - m)
		golomb_result = ""
		for val in tqdm(vals,desc="Golomb"):
			val = valueToSymbol[val]
			#golomb_result += toGolomb(val, m, c, div) #15.000/s
			golomb_result += Golomb.to_golomb(val, m, c, div) #14.500/s
		fhandler.write(golomb_result.encode("utf-8"))
		

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

def dumpFrames(shared_dict, handler, threshold = 50):
	if len(shared_dict)<threshold:
		return
	
	y_overall = []
	u_overall = []
	v_overall = []
	for frame in shared_dict.keys():
		y,u,v = shared_dict[frame]
		y_overall += y
		u_overall += u
		v_overall += v

	shared_dict.clear()

	values = (y_overall,u_overall,v_overall)

	processes = []
	m_proc = []
	symbolToValue_proc = []
	valueToSymbol_proc = []
	mgr = Manager()
	
	for v in values:
		m = Value('d',0.0)
		symbolToValue = mgr.dict()
		valueToSymbol = mgr.dict()
		processes.append(Process(target=processChannel,args=(shared_dict,v,m,symbolToValue,valueToSymbol)))
		m_proc.append(m)
		symbolToValue_proc.append(symbolToValue)
		valueToSymbol_proc.append(valueToSymbol)
	
	for p in processes:
		p.start()
	for p in processes:
		p.join()

	sym = []
	m = []
	for i in range(len(symbolToValue_proc)):
		s = {}
		# Removing probabilities from symbolToValue map. valueToSymbol already doesn't have them.
		for k in symbolToValue_proc[i].keys():
			s.update({k:symbolToValue_proc[i][k][0]})
		sym.append((s,valueToSymbol_proc[i]))
		m.append(m_proc[i].value)

	encodeValues(handler, values, sym, m)

def processChannel(shared_dict, overall,return_m,return_symbolToValue,return_valueToSymbol):
	#### Find best 'm' for every value in each overall
	# Calculate probabilty of each sample value after predictor is applied
	sample_probability = GolombEst.prob(overall)
	
	# Map symbols to sample values according to probability
	#{ symbol: (value, probability)}
	symbolToValue, valueToSymbol  = GolombEst.map_symbols(sample_probability)

	# Find best Golomb Parameters
	alpha, m = GolombEst.findBestGolomb(symbolToValue,False)

	# Return to shared variables
	return_m.value = m
	for k in symbolToValue:
		return_symbolToValue[k] = symbolToValue[k]
	for k in valueToSymbol:
		return_valueToSymbol[k] = valueToSymbol[k]

def decode(inputFile, outputFile):
	return 0

if __name__ == "__main__":
	main()
