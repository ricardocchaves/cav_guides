import codec_1 as c
import player as p
from multiprocessing import Process, Value, Manager, cpu_count

from VideoCaptureYUV import VideoCaptureYUV
from golomb_cython import ReadGolomb
from pixel_iteration import decodeFrame

import sys
from unittest.mock import patch
from tqdm import tqdm
import numpy as np

def encode(inputVid, encoded):
	cap = VideoCaptureYUV(inputVid)

	header = cap.header.encode('utf-8')

	output = open(encoded, 'wb')
	output.write(header)

	cnt = 1 # number of frames so far
	manager = Manager()
	shared_dict = manager.dict()
	for _ in range(20):
		ret, frame = cap.read_raw()
		if ret:
			c.processFrame(frame,cnt,shared_dict)
			c.dumpFrames(shared_dict,output,5)
			cnt += 1

	#c.dumpFrames(shared_dict,output,0)

def main():
	inputVid = "../videos/ducks_take_off_420_720p50.y4m"
	encoded = "trash_random_file.bin"
	outputVid = "trash_random_file.y4m"

	print("Encoding...")
	encode(inputVid, encoded)

	print("Decoding...")
	c.decode(encoded, outputVid)

	"""

	print("Playing original")
	testargs = ["prog", inputVid]
	with patch.object(sys, 'argv', testargs):
			p.main()

	print("Playing decoded")
	testargs = ["prog", outputVid]
	with patch.object(sys, 'argv', testargs):
			p.main()
	"""

if __name__ == "__main__":
	main()