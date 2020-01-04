import sys
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Value, Manager, cpu_count
from bitstring import BitArray
import time

from VideoCaptureYUV import VideoCaptureYUV

import sys
sys.path.append('..') # Making the other packages in the repository visible.

from proj2.Golomb import Golomb
import proj2.encode as GolombEst

from GolombFast import GolombFast

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
				"""
				if len(shared_dict.keys())>60:
					writeFrames(shared_dict,output)
				"""

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
	#writeFrames(shared_dict,output)
	dumpFrames(shared_dict,output,0)

def processFrame(height,width,frame,ID,shared_dict):
	#y_overall, u_overall, v_overall = pixel_iteration_slow(height,width,frame)
	#y_overall,u_overall,v_overall = pixel_iteration_fast(height,width,frame)
	"""
	overalls = pixel_iteration_fast(height,width,frame)
	l = len(overalls)//3
	y_overall = overalls[:l]
	u_overall = overalls[l:l*2]
	v_overall = overalls[l*2:]
	"""
	# Add y,u,v overall to globalOverall (y,u and v) dict/list in the frame position
	y_overall,u_overall,v_overall = pixel_iteration_np(frame,width,height)
	#overall = pixel_iteration_np(frame,width,height)
	shared_dict[ID] = (y_overall.flatten().tolist(),u_overall.flatten().tolist(),v_overall.flatten().tolist())
	#shared_dict[ID] = overall

def pixel_iteration_np(frame,width,height):
	# prediction = np.zeros([height,width,3],dtype=int)

	"""
	predictor_vec = np.vectorize(predictor,excluded=['frame'])
	x = np.arange(width).astype(int)
	for y in range(height):
		p = predictor_vec(x=x,y=y,frame=frame.astype(int))
		prediction[x,y] = p
	"""
	
	"""
	f = frame.copy()
	it = np.nditer(f, flags=['multi_index','c_index','reduce_ok'], op_flags=[['readonly']])
	while not it.finished:
		x,y,_ = it.multi_index
		if x >= width or y >= height:
			it.iternext()
			continue
		if((prediction[x,y]==np.zeros(3)).all()):
			prediction[x,y] = predictor(x,y,frame)
		it.iternext()
	"""

	predicted = predictor(frame)
	#return frame-predicted
	return np.dsplit(frame - predicted,3)
	#y_p,u_p,v_p = np.dsplit(predicted,3)
	#y,u,v = np.dsplit(frame,3)
	#y_p,u_p,v_p = np.dsplit(np.apply_along_axis(predictor,3,predicted_frame),3)
	#np.apply_along_axis(f,3,frame)
	#return [],[],[]

def predictor(frame):
	a = np.roll(frame,1,axis=1)
	# 1st column of a -> zeros
	a[:,0] = np.zeros((a.shape[0],a.shape[2]))
	
	b = np.roll(frame,1,axis=0)
	# 1st row of b -> zeros
	b[0] = np.zeros(b.shape[1:])
	
	c = np.roll(frame,[1,1],axis=(0,1))
	# 1st row and column -> zeros
	c[:,0] = np.zeros((c.shape[0],c.shape[2]))
	c[0] = np.zeros(c.shape[1:])

	f = np.frompyfunc(_nonLinearPredictor_np,5,1)
	max_ab = np.maximum(a,b)
	min_ab = np.minimum(a,b)
	return f(a,b,c,max_ab,min_ab)


def pixel_iteration_slow(height,width,frame):
	overalls = ([],[],[]) # y,u,v
	for h in range(height):
		for w in range(width):
			y,u,v = frame[h,w]
			y_p, u_p, v_p = _predictor(frame,h,w)

			diff_y = y - y_p
			diff_u = u - u_p
			diff_v = v - v_p

			overalls[0].append(diff_y)
			overalls[1].append(diff_u)
			overalls[2].append(diff_v)
	return overalls

def _predictor(frame,pos_h,pos_w):
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

def _nonLinearPredictor_np(a,b,c,maxAB,minAB):
	if c >= maxAB:
		return minAB
	elif c <= minAB:
		return maxAB
	else:
		return a+b-c

# c | b
# a | X
def _nonLinearPredictor(a,b,c):
	if c >= max(a,b):
		return min(a,b)
	elif c <= min(a,b):
		return max(a,b)
	else:
		return a+b-c

def writeFrames(shared_dict, handler):
	if len(shared_dict.keys()) == 0:
		return
	y_overall = []
	u_overall = []
	v_overall = []
	for frame in shared_dict.keys():
		y,u,v = np.dsplit(shared_dict[frame],3)
		y_overall += y.flatten().tolist()
		u_overall += u.flatten().tolist()
		v_overall += v.flatten().tolist()

	shared_dict.clear()

	values = (y_overall,u_overall,v_overall)

	handler.write(b"\DATA\n")

	for ch in range(len(values)):
		vals = values[ch]
		m = 8
		c = int(math.ceil(math.log(m,2)))
		div = int(math.pow(2,c) - m)

		golomb_result = BitArray()

		for v in vals:
			if v <0:
				g = '0'+Golomb.to_golomb_fast(abs(v), m, c, div)
			else:
				g = Golomb.to_golomb_fast(v, m, c, div)
			golomb_result += BitArray(bin=g)

		handler.write(golomb_result.tobytes())

def dumpFrames(shared_dict, handler, threshold = 50):
	if len(shared_dict)<threshold:
		return

	"""
	diffs = np.empty((0,720,1280,3),int)
	for frame in shared_dict.keys():
		diffs = np.append(diffs,[shared_dict[frame]],axis=0)
		shared_dict[frame] = []
	"""

	y_overall = []
	u_overall = []
	v_overall = []
	#np.empty((720,1280,3), int)
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
	metadata = {}
	for ch in range(len(sym)):
		symbolToValue,_ = sym[ch]
		metadata[ch] = symbolToValue
		
	from pickle import dumps
	metadata = dumps(metadata) # bytes
	fhandler.write(metadata)

	fhandler.write(b"\nDATA\n")

	for ch in range(len(values)):
		_,valueToSymbol_old = sym[ch]
		valueToSymbol = valueToSymbol_old.copy()
		vals = values[ch]
		m = int(m_list[ch])
		c = int(math.ceil(math.log(m,2)))
		div = int(math.pow(2,c) - m)
		#golomb_result = ""
		#golomb_result = []
		#print("Starting golomb")
		#s = time.time()
		golomb = GolombFast(m,c,div)
		for v in vals:
			golomb.toGolomb(valueToSymbol[v])
		#print("Time: {}".format(time.time()-s))
		"""
		s = time.time()
		symbols = np.array([valueToSymbol[v] for v in vals])
		print("Creating symbols: {:.2f}".format(time.time()-s))
		s = time.time()
		golomb_vect = np.vectorize(Golomb.to_golomb_fast,excluded=['m','c','div'])
		golomb_result = golomb_vect(val=symbols,m=m,c=c,div=div).tolist()
		print("Golomb vectorized: {:.2f}".format(time.time()-s))
		
		s = time.time()
		for i in range(len(vals)):

			symbol = valueToSymbol[vals[i]]
			#golomb_result += toGolomb_fast(symbol, m, c, div)
			#golomb_result += Golomb.to_golomb(symbol, m, c, div) #14.500/s
			g = Golomb.to_golomb_fast(symbol, m, c, div)
			golomb_result += g
			print(len(g))
			sys.exit(0)
			#print("val: {}, symbol:{}, golomb len:{}, m:{}, c:{}, div:{}".format(vals[i],symbol,len(golomb_result),m,c,div))
		print("Golomb for loop: {:.2f}".format(time.time()-s))
		sys.exit(0)
		"""
		#fhandler.write(golomb_result.encode('utf-8'))
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

def decode(inputFile, outputFile):
	return 0

if __name__ == "__main__":
	main()
