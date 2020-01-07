import sys
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Value, Manager, cpu_count
import time

from VideoCaptureYUV import VideoCaptureYUV

import sys
sys.path.append('..') # Making the other packages in the repository visible.

import proj2.encode as GolombEst

from GolombFast import GolombFast
from golomb_cython import GolombCython
from golomb_cython import ReadGolomb
from pixel_iteration import decodeFrame

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

	cnt = 1 # number of frames so far
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

			p = Process(target=processFrame,args=(frame,cnt,shared_dict))
			p.start()
			pbar.update(1)

			processes[currentProcess] = p
			cnt += 1
			currentProcess += 1
		else:
			break
	
	# In case there are remaining processes outside of a "batch"
	for p in processes:
		processes[p].join()

	dumpFrames(shared_dict,output,0)

def processFrame(frame,ID,shared_dict):
	# Add y,u,v overall to globalOverall (y,u and v) dict/list in the frame position
	y_overall = pixel_iteration_np(frame.getMatrix(0),frame.width,frame.height)
	u_overall = pixel_iteration_np(frame.getMatrix(1),frame.width_chroma,frame.height_chroma)
	v_overall = pixel_iteration_np(frame.getMatrix(2),frame.width_chroma,frame.height_chroma)
	#y_overall = pixel_iteration(frame.getMatrix(0),frame.width,frame.height)
	#u_overall = pixel_iteration(frame.getMatrix(1),frame.width_chroma,frame.height_chroma)
	#v_overall = pixel_iteration(frame.getMatrix(2),frame.width_chroma,frame.height_chroma)
	#print(y_overall[:2])
	#print(u_overall[:2])
	#print(v_overall[:2])

	shared_dict[ID] = (y_overall.flatten().tolist(),u_overall.flatten().tolist(),v_overall.flatten().tolist())

def pixel_iteration_np(frame,width,height):
	predicted = predictor(frame)
	return frame - predicted

# c | b
# a | X
def predictor(frame):
	a = np.roll(frame,1,axis=1)
	# 1st column of a -> zeros
	a[:,0] = np.zeros((a.shape[0]))

	b = np.roll(frame,1,axis=0)
	# 1st row of b -> zeros
	b[0] = np.zeros(b.shape[1])

	c = np.roll(frame,[1,1],axis=(0,1))
	# 1st row and column -> zeros
	c[:,0] = np.zeros((c.shape[0]))
	c[0] = np.zeros(c.shape[1])

	f = np.frompyfunc(_nonLinearPredictor_np,5,1)
	max_ab = np.maximum(a,b)
	min_ab = np.minimum(a,b)
	ret = f(a,b,c,max_ab,min_ab).astype(int)
	return ret

def pixel_iteration(frame,width,height):
	predicted = np.zeros((height,width),dtype=int)
	for y in range(height):
		for x in range(width):
			predicted[y,x] = predictorSlow(x,y,frame)
	
	return frame - predicted

def predictorSlow(x,y,frame):
	if x>0:
		a = int(frame[y,x-1])
	else:
		a = 0
	if y>0:
		b = int(frame[y-1,x])
	else:
		b = 0
	if y>0 and x>0:
		c = int(frame[y-1,x-1])
	else:
		c = 0
	
	return _nonLinearPredictor(a,b,c)

def _nonLinearPredictor_np(a,b,c,maxAB,minAB):
	if c >= maxAB:
		return minAB
	elif c <= minAB:
		return maxAB
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
		processes.append(Process(target=processChannel,args=(v,m,symbolToValue,valueToSymbol)))
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

def encodeValues(fhandler,values, sym, m_list):
	# Writing metadata
	fhandler.write(b"\nMETA\n")
	metadata = {}
	for ch in range(len(sym)):
		symbolToValue,_ = sym[ch]
		metadata[ch] = (m_list[ch],symbolToValue.copy())

	from pickle import dumps
	metadata = dumps(metadata) # bytes
	fhandler.write(metadata)

	fhandler.write(b"\nDATA\n")

	for ch in range(len(values)):
		fhandler.write(b"CHANNEL\n")
		_,valueToSymbol_old = sym[ch]
		valueToSymbol = valueToSymbol_old.copy()
		vals = values[ch]
		m = int(m_list[ch])
		c = int(math.ceil(math.log(m,2)))
		div = int(math.pow(2,c) - m)
		#golomb_result = ""
		#golomb_result = []
		golomb = GolombCython(m,c,div)
		for v in vals:
			golomb.toGolomb(valueToSymbol[v])

		fhandler.write(golomb.getBytes())

def processChannel(overall,return_m,return_symbolToValue,return_valueToSymbol):
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

## Decoding Part

def decode(inputFile, outputFile):
	fout = open(outputFile, 'wb')
	src = VideoCaptureYUV(inputFile)
	src.f.seek(0)

	fout.write(src.header.encode('utf-8'))

	print("Reading raw file to memory... ",end='')
	data_raw = src.f.read() # Read everything to memory
	print("Done")
	isData = True
	
	meta_start = -1
	meta_tag = b'\nMETA\n'
	meta_len = len(meta_tag)
	
	data_start = -1
	data_tag = b'\nDATA\n'
	data_len = len(data_tag)

	ch_start = -1
	ch_tag = b'CHANNEL\n'
	ch_len = len(ch_tag)
	
	next_meta = -1
	next_ch = -1
	
	while isData:
		# Estabilishing limits of current "batch"
		print("#### Setting limits of batch...")
		meta_start = data_raw.find(meta_tag, meta_start+1)
		data_start = data_raw.find(data_tag, data_start+1)
		next_meta = data_raw.find(meta_tag, meta_start+1)
		metadata = data_raw[meta_start+meta_len:data_start]
		if next_meta != -1:
			data = data_raw[data_start+data_len:next_meta]
		else:
			data = data_raw[data_start+data_len:]
			isData = False

		# Extracting metadata
		print("Extracting metadata...")
		from pickle import loads
		metadata_tmp = loads(metadata) # metadata dict only has 3 keys: 0,1,2 - one per channel
		metadata = {}
		for ch in range(3):
			metadata[ch]=metadata_tmp[ch]
			del(metadata_tmp[ch])

		# Extracting data
		print("Extracting data...")
		overalls = []
		ch_start = -1
		next_ch = -1
		for ch in tqdm(range(3),desc='Reading channel vals'):
			m, symToVal = metadata[ch]
			ch_start = data.find(ch_tag, ch_start+1)
			next_ch = data.find(ch_tag, ch_start+1)
			if next_ch != -1:
				ch_data_bytes = data[ch_start+ch_len:next_ch]
			else:
				ch_data_bytes = data[ch_start+ch_len:]

			# Decode channel data
			g = ReadGolomb(m,ch_data_bytes)
			values_sym = g.getValues()
			values = [symToVal[v] for v in values_sym] # Original values
			overalls.append(values)
		
		f_w = 1280
		f_h = 720
		frameSize = f_w*f_h #TODO generic
		numberOfFrames = int(len(overalls[0])/frameSize) #TODO generic
		for frame in tqdm(range(numberOfFrames),desc='Processing batch'):
			fout.write(b'FRAME\n')
			mgr = Manager()
			decoded_bytes = mgr.dict()
			"""
			for ch in range(3):
				ch_diff = np.array(overalls[ch][frame*frameSize:(frame+1)*frameSize]).reshape((f_h,f_w))
				decoded_data = decodeFrame(f_h,f_w,ch_diff).astype(np.uint8)
				decoded_data_bytes = decoded_data.flatten().tobytes()
				#print(decoded_data.shape, len(decoded_data_bytes))
				fout.write(decoded_data_bytes)
			"""
			procs = []
			for ch in range(3):
				ch_diff = np.array(overalls[ch][frame*frameSize:(frame+1)*frameSize]).reshape((f_h,f_w))
				procs.append(Process(target=decodeChannelProcess,args=(f_h,f_w,ch_diff,ch,decoded_bytes)))
				procs[ch].start()
			for p in procs:
				p.join()
				p.terminate()
			for ch in range(3):
				fout.write(decoded_bytes[ch])

def decodeChannelProcess(f_h,f_w,ch_diff,ch,return_bytes):
	decoded_data = decodeFrame(f_h,f_w,ch_diff).astype(np.uint8)
	decoded_data_bytes = decoded_data.flatten().tobytes()
	return_bytes[ch] = decoded_data_bytes

def decode_predictor(x,y,frame):
	if x>0:
		a = int(frame[y,x-1])
	else:
		a = 0
	if y>0:
		b = int(frame[y-1,x])
	else:
		b = 0
	if y>0 and x>0:
		c = int(frame[y-1,x-1])
	else:
		c = 0
	
	return _nonLinearPredictor(a,b,c)

def _nonLinearPredictor(a,b,c):
	maxAB = max(a,b)
	minAB = min(a,b)
	if c >= maxAB:
		return minAB
	elif c <= minAB:
		return maxAB
	else:
		return a+b-c

if __name__ == "__main__":
	main()
