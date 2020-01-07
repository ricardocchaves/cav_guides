import sys
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Value, Manager, cpu_count
import time

from VideoCaptureYUV import VideoCaptureYUV, Frame

import sys
sys.path.append('..') # Making the other packages in the repository visible.

import proj2.encode as GolombEst

from GolombFast import GolombFast
from golomb_cython import GolombCython
from golomb_cython import ReadGolomb
from pixel_iteration import decodeFrame

import inter_frame_coding_fast as inter

def main():
	if(len(sys.argv) <= 3):
		print("Lossy hybrid encoder (intra+inter coding)")
		print("USAGE: python3 {} inputFile outputFile [args]".format(sys.argv[0]))
		print("Args:")
		print("    -encode")
		print("    -decode")
		print("	   -n n_value")
		print("    -off offset")
		print("	   -q value ")
		return

	inputFile = sys.argv[1]
	outFile = sys.argv[2]
	N = 8
	offset = 2
	quantization_steps = 2
	for arg in sys.argv[3:]:
		if arg == "-encode":
			encode(inputFile,outFile,N,offset,quantization_steps)
		elif arg == "-decode":
			decode(inputFile,outFile)
		elif '-n' in arg:
			N = int(arg.split(' ')[1])
		elif '-off' in arg:
			offset = int(arg.split(' ')[1])
		elif '-q' in arg:
			quantization_steps = 2**int(arg.split(' ')[1])

def encode(inputFile, outputFile, N, offset, quantization_steps):
	cap = VideoCaptureYUV(inputFile)

	header = cap.header.encode('utf-8')

	output = open(outputFile, 'wb')
	output.write(header)

	frames_dict = {}

	ref_frame = None

	cnt = 1 # number of frames so far
	pbar = tqdm(total=500)

	while True:
		ret, frame = cap.read_raw()
		if ret:
			# If ref_frame doesn't exist
			if not ref_frame:
				# Start of batch, current frame will be reference
				frames_dict[cnt] = processFrame(frame,quantization_steps)
				ref_frame = frame
			else:
				motion_vectors_to_encode, residual_diffs = inter.frame_processing(cap.height,cap.width,cap.height_chroma,cap.width_chroma,frame, ref_frame, N, offset)

				mv_aslist = inter.mv_tolist(motion_vectors_to_encode,cap.height,cap.width,N)

				residuals_aslist = inter.residual_diffs_tolist(residual_diffs,cap.height,cap.width,N)

				frames_dict[cnt] = (mv_aslist,residuals_aslist)
			
			wroteFrames = dumpFrames(frames_dict,output,N=N,offset=offset,q=quantization_steps)
			if wroteFrames:
				ref_frame = None
				frames_dict.clear()
			
			pbar.update(1)
			cnt += 1
		else:
			break

	dumpFrames(frames_dict,output,N=N,offset=offset,q=quantization_steps)

def processFrame(frame, quantization_steps):
	# Add y,u,v overall to globalOverall (y,u and v) dict/list in the frame position
	y_overall = pixel_iteration_np(frame.getMatrix(0),frame.width,frame.height)
	u_overall = pixel_iteration_np(frame.getMatrix(1),frame.width_chroma,frame.height_chroma)
	v_overall = pixel_iteration_np(frame.getMatrix(2),frame.width_chroma,frame.height_chroma)

	# Quantization
	S = math.ceil(256/quantization_steps)
	y_overall = y_overall//S
	u_overall = u_overall//S
	v_overall = v_overall//S

	return (y_overall.flatten().tolist(),u_overall.flatten().tolist(),v_overall.flatten().tolist())

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

def dumpFrames(shared_dict, handler, threshold = 50, N=8, offset=2, q=0):
	if len(shared_dict)<threshold:
		return False

	first = True
	for frame in shared_dict.keys():
		if first:
			# Reference frame is intra coded
			first = False

			y,u,v = shared_dict[frame]
			values = (y,u,v)
			sym_list = []
			m_list = []

			m,symbolToValue,valueToSymbol = processChannel(y)
			s = {}
			for k in symbolToValue.keys():
				s.update({k:symbolToValue[k][0]})
			sym = (s,valueToSymbol)
			sym_list.append(sym)
			m_list.append(m)

			m,symbolToValue,valueToSymbol = processChannel(u)
			s = {}
			for k in symbolToValue.keys():
				s.update({k:symbolToValue[k][0]})
			sym = (s,valueToSymbol)
			sym_list.append(sym)
			m_list.append(m)

			m,symbolToValue,valueToSymbol = processChannel(v)
			s = {}
			for k in symbolToValue.keys():
				s.update({k:symbolToValue[k][0]})
			sym = (s,valueToSymbol)
			sym_list.append(sym)
			m_list.append(m)

			encodeValues(handler, values, sym_list, m_list,q)
		else:
			mv_aslist,residuals_aslist = shared_dict[frame]
			values = mv_aslist,residuals_aslist

			m, symbolToValue, valueToSymbol = processRefFrame(mv_aslist,residuals_aslist)

			s = {}
			# Removing probabilities from symbolToValue map. valueToSymbol already doesn't have them.
			for k in symbolToValue.keys():
				s.update({k:symbolToValue[k][0]})
			#encode interencoded frame
			encodeRefFrame(handler, values, s, m, valueToSymbol,N,offset)

	return True

def encodeValues(fhandler,values, sym, m_list,q):
	# Writing metadata
	fhandler.write(b"\nMETA\n")
	metadata = {}
	for ch in range(len(sym)):
		symbolToValue,_ = sym[ch]
		metadata[ch] = (m_list[ch],symbolToValue,q)

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

def encodeRefFrame(fhandler,values, sym, m, valueToSymbol,N,offset):
	# Writing metadata
	fhandler.write(b"\nMETA\n")
	mv,residuals=values
	metadata = {}
	metadata['mv_len'] = len(mv)
	metadata['residuals_len'] = len(residuals)
	metadata['m'] = m
	metadata['symbolToValue'] = sym
	metadata['N'] = N
	metadata['offset'] = offset

	from pickle import dumps
	metadata = dumps(metadata) # bytes
	fhandler.write(metadata)

	fhandler.write(b"\nDATA\n")
	
	c = int(math.ceil(math.log(m,2)))
	div = int(math.pow(2,c) - m)
	#golomb_result = ""
	#golomb_result = []
	golomb = GolombCython(m,c,div)
	for v in mv+residuals:
		golomb.toGolomb(valueToSymbol[v])

	fhandler.write(golomb.getBytes())

def processRefFrame(motion_vectors,residuals):
	#### Find best 'm' for every value in each overall
	# Calculate probabilty of each sample value after predictor is applied
	overall = motion_vectors+residuals
	sample_probability = GolombEst.prob(overall)

	# Map symbols to sample values according to probability
	#{ symbol: (value, probability)}
	symbolToValue, valueToSymbol  = GolombEst.map_symbols(sample_probability)

	# Find best Golomb Parameters
	alpha, m = GolombEst.findBestGolomb(symbolToValue,False)

	return m,symbolToValue,valueToSymbol

def processChannel(y,u=None,v=None):
	#### Find best 'm' for every value in each overall
	# Calculate probabilty of each sample value after predictor is applied
	if u==None and v==None:
		overall = y
	else:
		overall = y+u+v
	sample_probability = GolombEst.prob(overall)

	# Map symbols to sample values according to probability
	#{ symbol: (value, probability)}
	symbolToValue, valueToSymbol  = GolombEst.map_symbols(sample_probability)

	# Find best Golomb Parameters
	alpha, m = GolombEst.findBestGolomb(symbolToValue,False)

	return m,symbolToValue,valueToSymbol

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

	ref_frame = None
	
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
		metadata_tmp = loads(metadata) 
		if len(metadata_tmp.keys()) > 3:
			# Inter frame METADATA
			metadata = {}
			mv_len = metadata['mv_len']
			residuals_len = metadata['residuals_len']
			m = metadata['m']
			sym = metadata['symbolToValue']
			N = metadata['N']
			offset = metadata['offset']

			# Extracting data
			print("Extracting data...")
			ch_start = data.find(ch_tag, ch_start+1)
			next_ch = data.find(ch_tag, ch_start+1)
			if next_ch != -1:
				ch_data_bytes = data[ch_start+ch_len:next_ch]
			else:
				ch_data_bytes = data[ch_start+ch_len:]

			# Decode channel data
			g = ReadGolomb(m,ch_data_bytes)
			values_sym = g.getValues()
			values = [sym[v] for v in values_sym] # Original values
			
			motion_vectors = values[:mv_len]
			residuals = values[mv_len:residuals_len]

			mv_to_decode = inter.mv_fromlist(motion_vectors,src.height,src.width,N)
			residuals_to_decode = inter.residuals_fromlist(residuals,src.height,src.width,src.height_chroma, src.width_chroma,N)

			decoded_frame = inter.decode_inter_frame(src.height,src.width,src.height_chroma,src.width_chroma,ref_frame,N, mv_to_decode, residuals_to_decode)
    
			fout.write(b'FRAME\n')

			for ch in range(3):
				fout.write(decoded_frame.getMatrix(ch).flatten().tobytes())
		else:
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
				m, symToVal,N = metadata[ch]
				S = math.ceil(256/N)
				ch_start = data.find(ch_tag, ch_start+1)
				next_ch = data.find(ch_tag, ch_start+1)
				if next_ch != -1:
					ch_data_bytes = data[ch_start+ch_len:next_ch]
				else:
					ch_data_bytes = data[ch_start+ch_len:]

				# Decode channel data
				g = ReadGolomb(m,ch_data_bytes)
				values_sym = g.getValues()
				values = [symToVal[v]*S for v in values_sym] # Original values
				overalls.append(values)
			
			frameSize = src.height*src.width
			numberOfFrames = 1
			for frame in tqdm(range(numberOfFrames),desc='Processing batch'):
				fout.write(b'FRAME\n')
				mgr = Manager()
				decoded_bytes = mgr.dict()

				procs = []
				for ch in range(3):
					if ch == 0:
						f_h = src.height
						f_w = src.width
					else:
						f_h = src.height_chroma
						f_w = src.width_chroma
					frameSize = f_h*f_w
					ch_diff = np.array(overalls[ch][frame*frameSize:(frame+1)*frameSize]).reshape((f_h,f_w))
					procs.append(Process(target=decodeChannelProcess,args=(f_h,f_w,ch_diff,ch,decoded_bytes)))
					procs[ch].start()
				for p in procs:
					p.join()
					p.terminate()
				tmp_ref_frame = []
				for ch in range(3):
					tmp_ref_frame.append(decoded_bytes[ch])
					fout.write(decoded_bytes[ch])
				y = np.frombuffer(tmp_ref_frame[0], dtype=np.uint8)
				u = np.frombuffer(tmp_ref_frame[1], dtype=np.uint8)
				v = np.frombuffer(tmp_ref_frame[2], dtype=np.uint8)
				ref_frame = Frame(y,u,v, src.height, src.height_chroma, src.width, src.width_chroma)
				

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
