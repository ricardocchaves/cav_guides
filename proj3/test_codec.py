import codec_3 as c
import player as p
import inter_frame_coding_fast as inter
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

	quantization_steps = 2

	header = cap.header.encode('utf-8')

	output = open(encoded, 'wb')
	output.write(header)

	ref_frame = None
	frames_dict = {}

	N = 8
	offset = 2

	cnt = 8 # number of frames so far
	manager = Manager()
	shared_dict = manager.dict()
	for _ in range(20):
		ret, frame = cap.read_raw()
		if ret:
			if not ref_frame:
				# Start of batch, current frame will be reference
				frames_dict[cnt] = c.processFrame(frame,quantization_steps)
				ref_frame = frame
			else:
				motion_vectors_to_encode, residual_diffs = inter.frame_processing(cap.height,cap.width,cap.height_chroma,cap.width_chroma,frame, ref_frame, N, offset)

				mv_aslist = inter.mv_tolist(motion_vectors_to_encode,cap.height,cap.width,N)

				residuals_aslist = inter.residual_diffs_tolist(residual_diffs,cap.height,cap.width,N)

				frames_dict[cnt] = (mv_aslist,residuals_aslist)
			
			wroteFrames = c.dumpFrames(frames_dict,output,5,N=N,offset=offset)
			if wroteFrames:
				ref_frame = None
				frames_dict.clear()

			cnt += 1

	c.dumpFrames(shared_dict,output,0)

def main():
	inputVid = "../videos/ducks_take_off_444_720p50.y4m"
	encoded = "trash_random_file.bin"
	outputVid = "trash_random_file.y4m"

	print("Encoding...")
	encode(inputVid, encoded)

	print("Decoding...")
	c.decode(encoded, outputVid)

	print("Playing original")
	testargs = ["prog", inputVid]
	with patch.object(sys, 'argv', testargs):
			p.main()

	print("Playing decoded")
	testargs = ["prog", outputVid]
	with patch.object(sys, 'argv', testargs):
			p.main()

if __name__ == "__main__":
	main()